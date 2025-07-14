import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage

# 🔐 Load secrets (supports local `.env`)
load_dotenv()
openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets or a .env file.")
    st.stop()


# 🚀 Streamlit UI
st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A App using LangChain + Streamlit")

# 📥 File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# 🧠 PDF Text Extraction
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# 📚 Text Splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# 🧱 Embedding + Vectorstore
def embed_and_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    return Chroma.from_texts(docs, embedding=embeddings)

# 🎯 Answer Questions
def answer_query(query, db):
    llm = ChatOpenAI(openai_api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa_chain.run(query)

# 🧪 Main App Logic
if uploaded_file:
    with st.spinner("📄 Extracting and processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(raw_text)
        db = embed_and_store(chunks)
        st.success("✅ PDF processed and vectorized!")

    st.markdown("---")
    st.subheader("🔍 Ask a question about the PDF")
    query = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if query:
            with st.spinner("🤔 Thinking..."):
                response = answer_query(query, db)
                st.markdown("**Answer:** " + response)
        else:
            st.warning("Please enter a question first.")
