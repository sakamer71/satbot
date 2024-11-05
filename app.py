import openai
openai.log = "debug"

import streamlit as st
from streamlit import session_state as ss
from openai import AzureOpenAI
import os
import json
import pickle
import time
import glob
import boto3
import sys
import logging

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
        logging.FileHandler("/var/log/streamlit/streamlit.log"),
        logging.StreamHandler()  # Optional: to also print logs to console
    ]
)

# openai.logger.setLevel(logging.DEBUG)

logger = logging.getLogger('streamlit')

starter_prompts = [
    "Hello, how can I help you today?",
    "What are your thoughts on AI?",
    "Can you tell me more about your project?",
]

st.sidebar.title("Starter Prompts")
selected_prompt = st.sidebar.radio("Select a prompt:", starter_prompts)

# Redirect stdout and stderr to logger
class StreamToLogger:
    def __init__(self, log_level):
        self.log_level = log_level

    def write(self, message):
        if message.strip():
            self.log_level(message)

    def flush(self):
        pass

# Redirect stdout and stderr
sys.stdout = StreamToLogger(logger.info)
sys.stderr = StreamToLogger(logger.error)


#from latexgen import latex2image

## get parameter from ssm parameter store
def get_ssm_parameter(parameter):
    result = ss.ssm.get_parameter(Name=parameter)
    return result['Parameter']['Value']

# Azure OpenAI client setup
def initialize_openai_client(endpoint_parameter, api_key_parameter):

    client = AzureOpenAI(
        #azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_endpoint = get_ssm_parameter(endpoint_parameter),
        #api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_key = get_ssm_parameter(api_key_parameter),
        api_version="2024-09-01-preview"
    )
    ss['oaiclient'] = client

## establish aws client
def local_aws_client(service, region='us-east-2'):
    client = boto3.client(service, region_name=region)
    ss[service] = client
    print(client)
    return client


# Streamlit UI setup
def setup_streamlit():
    st.title("Cecibear Chatbot")
    st.write("Powered by the Azure OpenAI GPT-o1 model.")

    # Sidebar for saving and loading sessions
    st.sidebar.header("Previous Chat Sessions")
    #st.sidebar.write("The session will be auto-saved after the 2nd question is answered.")

    # Dropdown to select and load recent sessions
    saved_sessions = glob.glob("saved_sessions/*.pkl")
    saved_sessions = sorted(saved_sessions, reverse=True)  # Sort by most recent
    session_names = [os.path.basename(session) for session in sorted(saved_sessions, key=lambda x: ('topics_to_master' not in x, -os.path.getmtime(x)))]
    #selected_session = st.sidebar.selectbox("Load a previous session:", ["None"] + session_names)
    selected_session = st.sidebar.selectbox(label='Load a previous session', options=['None'] + session_names, format_func=lambda x: 'None' if x == 'None' else x.replace('_', ' ').split('.')[0])
    if selected_session != "None" and st.sidebar.button("Load Session"):
        load_session(selected_session)

# Initialize chat session
def initialize_chat():
    '''
    # initial_prompt = (
    #     "You excel at preparing students for SAT exams. When prompted for exam questions on a specific topic, "
    #     "you create questions and choices which are closely aligned with what will be tested on the SAT, and "
    #     "sometimes create more challenging questions or trickier options that a student should expect to see. "
    #     "You format all questions and choices with bold, colors, etc, so that they are easy to read. "
    #     "If the user requests math questions, the math equation portion of each question and choice should be "
    #     "formatted in raw LaTeX and surrounded by ${}. When user responds with their answers you evaluate whether "
    #     "the answer is correct, and for each choice you explain why the choice was correct or incorrect."
    # )
    '''
    initial_prompt = '''
    You excel at preparing students for SAT exams. When prompted for exam questions on a specific topic, you create questions and choices that are closely aligned with what will be tested on the SAT, including trickier options a student should expect to see.

    For math questions:
    - Format the math equations and expressions using LaTeX.
    - Ensure that any sections which have latex code are formatted such that streamlit st.write can properly render it. Placing latex inside double $$ is a good start.
    - Structure each question and choice as a combination of regular text and LaTeX, clearly distinguishing between them.
    - Use clear, readable text and LaTeX formatting that aligns with SAT standards.

    When a user responds with their answer, evaluate the correctness of the response and provide a detailed explanation for each choice, specifying why it is correct or incorrect.
    '''
    ##     - Surround all LaTeX code first with raw string quotes, and then with double dollar signs (`$$`) so that it is properly displayed in Streamlit.
    
    #ss["messages"] = [{"role": "user", "content": initial_prompt}]
    ss['messages'] = []
    #response = get_chat_response(initial_prompt)
    ss["question_count"] = 0

# Function to get response from Azure OpenAI
def get_chat_response(prompt):
    messages = ss.get("messages", [])
    response = ss.oaiclient.chat.completions.create(
        model="o1-preview",  # replace with the model deployment name of your o1-preview, or o1-mini model
        messages=messages + [{"role": "user", "content": prompt}],
        max_completion_tokens=5000
    )
    response_message = json.loads(response.model_dump_json())["choices"][0]["message"]["content"].strip()
    ss["messages"].append({"role": "assistant", "content": response_message})
    print(response_message)
    return response_message

# Save session to file
def save_session():
    # Summarize the first user question to create a session name
    #first_user_message = ss["messages"][2]["content"] if len(ss["messages"]) > 1 else ""
    first_user_message = ss["messages"][1]["content"] if len(ss["messages"]) > 0 else ""
    summary_prompt = f"Summarize the following question in 6 words or less with no punctuation: {first_user_message}"
    summary_response = get_chat_response(summary_prompt)
    session_name = summary_response.replace(" ", "_").replace("/", "-")[:50]  # Create a filename-friendly summary
    filename = f"saved_sessions/{session_name}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(ss.get("messages", []), f)
    st.sidebar.success(f"Session saved to {filename}")

# Load session from file
def load_session(filename):
    filepath = f"saved_sessions/{filename}"
    with open(filepath, "rb") as f:
        ss["messages"] = pickle.load(f)
    ss["initialized"] = True
    st.sidebar.success(f"Loaded session: {filename}")

# Display the entire chat history
def display_chat_history():
    for message in ss.get("messages", []):
        if message["role"] == "user":
            with st.chat_message('human', avatar='ceci.jpg'):
                st.write(f"{message['content']}")
        else:
            with st.chat_message('ai'):
                print('*****:AI RESPONSE****')
                print(message['content'])
                st.write(f"{message['content']}")
                #st.markdown(f"{message['content']}")

# Main function to manage Streamlit chatbot UI
def main():
    if 'ssm' not in ss:
        local_aws_client('ssm')

    # Initialize AzureOpenAI client
    initialize_openai_client('/satbot/api-endpoint/gpt-o1','/tce-helper/api-key/openai/gpt-o1')

    setup_streamlit()
    if "initialized" not in ss:
        initialize_chat()
        ss["initialized"] = True
    
    display_chat_history()

    # Accept new user input
    # Add a microphone button for speech input
    user_input = st.chat_input("Ask me: ")

    # Add a button to start speech recognition
    if st.button("🎤 Speak"):
        st.write("Listening...")

        # JavaScript code to capture speech
        st.markdown("""
<script>
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        if (event.results.length > 0) {
            const transcript = event.results[0][0].transcript;
            document.querySelector('input[type="text"]').value = transcript;
            document.querySelector('button[aria-label="🎤 Speak"]').innerText = "🎤 Speak";
        }
    };

    recognition.onspeechend = () => {
        console.log('Speech has stopped.');
        recognition.stop();
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error detected: ' + event.error);
        document.querySelector('button[aria-label="🎤 Speak"]').innerText = "🎤 Speak";
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
            alert('Please enable microphone permissions for this page.');
        }
    };

    document.querySelector('button[aria-label="🎤 Speak"]').onclick = () => {
        document.querySelector('button[aria-label="🎤 Speak"]').innerText = "Listening...";
        recognition.start();
    };
</script>

        """, unsafe_allow_html=True)
    if user_input:
        # Add user input to the message history and immediately display it
        ss["messages"].append({"role": "user", "content": user_input})
        ss["question_count"] += 1
        with st.chat_message('human', avatar='ceci.jpg'):
            st.write(f"{user_input}")
        
        with st.spinner(":blue[**JENGA TOWAH!!!!**] :red[**CovidyMcCovidFace :brain: :taco: :shocked_face_with_exploding_head: :shocked_face_with_exploding_head: is thinking...**]"):
            response = None
            retries = 5
            for attempt in range(retries):
                try:
                    response = get_chat_response(user_input)
                    break
                except Exception as e:
                    if "400" in str(e):
                        print(f"Error 400 occurred: {e}. Not retrying.")
                        st.error(f"Error 400 occurred: {e}.")
                        break
                    elif attempt < retries - 1:
                        print(f"Error occurred: {e}. Retrying ({attempt + 1}/{retries})...")
                        time.sleep(2)  # Optional: wait before retrying
                    else:
                        st.error("Failed to get a response from the AI after multiple attempts.")
                        raise e
        
        # Display the assistant's response
        with st.chat_message('ai'):
            st.write(f"{response}")
        
        # Save session after the 2nd question is answered
        if ss["question_count"] == 1:
            save_session()

    # Optional: display session state for debugging purposes
    with st.expander("Session State"):
        st.write(ss)

# Run the main function
if __name__ == "__main__":
    main()
