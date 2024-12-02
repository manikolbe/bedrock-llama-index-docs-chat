import boto3
import os
import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock
from llama_index.core import Settings


# Constants for model names and directories
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
CHROMA_DB_DIR = "./chroma_db"

Settings.embed_model = BedrockEmbedding(
    model_name=EMBED_MODEL,
    profile_name="default",
    region_name="us-east-1"
)

Settings.llm = Bedrock(
    model=LLM_MODEL,
    profile_name="default",
    region_name="us-east-1"
)

@st.cache_resource(show_spinner=False)
def get_index():
    # Load index from Chroma Vector Store
    index_path = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
    if os.path.exists(index_path):
        print("Chroma database found on disk. Loading the index...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=CHROMA_DB_DIR)
            index = load_index_from_storage(storage_context)
            print("Index loaded successfully.")
            return index
        except Exception as e:
            print(f"Failed to load the index: {e}")
    else:
        print("Index not found on disk.")
        os.system.exit(1)


st.subheader('RAG Using Amazon Bedrock Documentation', divider='rainbow')

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Amazon Bedrock service"}
    ]


index = get_index()
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
