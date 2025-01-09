import streamlit as st
from transformers import pipeline, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
import pickle

# Initialize QA model and embeddings
st.title("MHU QA System with URL Text Retrieval ")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize file paths
vector_store_path = "vectorstore.pkl"
docs_path = "docs.pkl"

# Extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()

        # Extract visible text
        text = soup.get_text(separator="\n").strip()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Sidebar for URL input
st.sidebar.title("URL Input")
url = st.sidebar.text_input("Enter a URL to process:")

if url:
    st.info("Extracting and processing text...")
    context = extract_text_from_url(url)
    if context:
        # Split the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([context])

        # Embed the text chunks
        vectorstore = FAISS.from_documents(docs, embedding_model)

        # Save the vector store and docs for later queries
        with open(vector_store_path, "wb") as f:
            pickle.dump(vectorstore, f)
        with open(docs_path, "wb") as f:
            pickle.dump(docs, f)

        st.success("Text processed and stored for fast retrieval!")

# Question Answering
st.subheader("Ask Questions")
question = st.text_input("Enter your question:")

if question:
    try:
        # Load vector store and documents
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        # Retrieve top-k relevant chunks
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question)

        # Combine relevant chunks into context
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Use the QA model
        result = qa_pipeline(question=question, context=context)
        st.header("Answer")
        st.write(result["answer"])
    except Exception as e:
        st.error(f"Error processing the question: {e}")
