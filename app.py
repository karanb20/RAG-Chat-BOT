import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
import os
import tempfile

# ✅ Load API Key
try:
    groq_key = st.secrets["groq_api_key"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    groq_key = os.getenv("groq_api_key")

# ✅ Page Config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Advanced RAG Chatbot")

# ✅ Sidebar
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.chain = None
    st.session_state.retriever = None
    st.rerun()


# ✅ Create Chain
@st.cache_resource
def create_chain(pdf_path):
    # Load & split
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings & vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant"
    )

    # Contextualize question using chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate the question to be standalone. "
         "Do NOT answer it, just rephrase if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the context below to answer "
         "the user's question concisely.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Format docs helper
    def format_docs(inputs):
        return "\n\n".join(doc.page_content for doc in inputs["context"])

    # Build RAG chain manually
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: history_aware_retriever.invoke({
                "input": x["input"],
                "chat_history": x.get("chat_history", [])
            })
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(format_docs)
        )
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, history_aware_retriever


# ✅ Process uploaded PDF
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    if "chain" not in st.session_state or st.session_state.chain is None:
        with st.spinner("⏳ Processing PDF..."):
            chain, retriever = create_chain(pdf_path)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.chat_history = []
        st.success("✅ PDF processed! Ask your questions.")

else:
    st.info("👈 Please upload a PDF from the sidebar to get started.")


# ✅ Init chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ✅ Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# ✅ Chat input
user_input = st.chat_input("Ask something about your document...")

if user_input:
    if not st.session_state.get("chain"):
        st.warning("⚠️ Please upload a PDF first.")
    else:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤔 Thinking..."):
            # Get source documents
            source_docs = st.session_state.retriever.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

            # Get answer
            answer = st.session_state.chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

        # Update chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            if source_docs:
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(source_docs):
                        page = doc.metadata.get('page', 0) + 1
                        st.markdown(f"**Source {i+1} — Page {page}:**")
                        st.write(doc.page_content[:300] + "...")