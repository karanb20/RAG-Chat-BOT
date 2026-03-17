from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("groq_api_key")

print("All libraries imported successfully")

# ✅ Load PDF
loader = PyPDFLoader('document.pdf')
documents = loader.load()

# ✅ Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# ✅ Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Vector store
vectorstore = FAISS.from_documents(docs, embeddings)
print("Vector database created successfully")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ LLM
llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.1-8b-instant"
)

def get_retriever():
    return retriever

def get_llm():
    return llm