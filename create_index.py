import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
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

# Load documents and build the index
documents = SimpleDirectoryReader("./data").load_data()

# Initialize Chroma client with persistent storage
print("Initializing Chroma client with persistent storage...")
db = chromadb.PersistentClient(path=CHROMA_DB_DIR)
chroma_collection = db.get_or_create_collection("quickstart")

# Initialize the vector store with the Chroma collection and embedding model
print("Initializing the vector store with the Chroma collection...")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# Save the index to disk
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
index.storage_context.persist(persist_dir=CHROMA_DB_DIR)
print("Index built and saved to disk.")