import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import tempfile
try:
    groq_key = st.secrets["groq_api_key"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    groq_key = os.getenv("groq_api_key")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Advanced RAG Chatbot")

st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

st.sidebar.markdown("---")
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.chain = None


@st.cache_resource
def create_chain(pdf_path):
    # Load & split
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings & vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")

    # Prompt to contextualize question using chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest user question, "
                   "reformulate the question to be standalone. "
                   "Do NOT answer it, just rephrase if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Prompt to answer the question
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context below to answer "
                   "the user's question concisely.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    return rag_chain


# Process uploaded PDF
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    if "chain" not in st.session_state or st.session_state.chain is None:
        with st.spinner("Processing PDF..."):
            st.session_state.chain = create_chain(pdf_path)
            st.session_state.chat_history = []
        st.success("PDF processed! Ask your questions.")

# Init chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Ask something about your document...")

if user_input:
    if not st.session_state.get("chain"):
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

        answer = response["answer"]
        source_docs = response.get("context", [])

        # Update chat history with LangChain message objects
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

        with st.chat_message("assistant"):
            st.markdown(answer)
            if source_docs:
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', '?')+1}):")
                        st.write(doc.page_content[:300] + "...")