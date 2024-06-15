import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms import OpenAI
import openai

# Streamlit App Configuration
st.set_page_config(
    page_title="Chat with your syllabus!", 
    page_icon="🦙", 
    layout="centered", 
    initial_sidebar_state="auto"
)

# Set OpenAI API Key
openai.api_key = st.secrets["openai_key"]

# Title and Information
st.title("Chat with the Syllabus and Course Outline 💬🦙")
st.info("Check out the Data Analytics Program at Miami Dade College [MDC](https://mdc.edu)", icon="📃")

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the course!"}
    ]

# Function to load and index course documents
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("Loading and indexing the course docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, 
            system_prompt=("You are a College Professor or Corporate Trainer and your job is to answer "
                           "questions about the course. Assume that all questions are related to the course and "
                           "documents provided. Keep your answers technical and based on facts – do not hallucinate features."))
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

# Load the data
index = load_data()

# Initialize the chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(st.session_state.messages[-1]["content"])
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
