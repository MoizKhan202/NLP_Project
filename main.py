import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader

# Initialize models
st.title("MHU News Research Tool 📈")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")

# Sidebar for URL input
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

# Process URLs
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    st.info("Loading data from URLs... Please wait!")
    try:
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        st.stop()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create embeddings
    st.info("Creating embeddings... Please wait!")
    docs_embeddings = [embedding_model.encode(doc.content) for doc in docs]
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    st.success("Data processed and embeddings saved!")

# Query input
query = st.text_input("Ask a question about the content:")
if query and file_path:
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    # Process query
    context = " ".join([doc.content for doc in docs])
    result = qa_pipeline(question=query, context=context)

    # Display answer
    st.header("Answer")
    st.write(result["answer"])
