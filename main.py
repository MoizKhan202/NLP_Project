import streamlit as st
import pickle
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Title and Sidebar
st.title("MHU: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Initialize Hugging Face models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
docs_path = "docs.pkl"  # Path to save the processed documents

def clean_text(text):
    """Clean and preprocess text."""
    return text.replace("\n", " ").strip()

if process_url_clicked:
    # Load data from URLs
    st.info("Loading data from URLs... Please wait!")
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        st.stop()

    # Split data into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=500,  # Smaller chunks for better context
        chunk_overlap=100  # Overlap to retain context across chunks
    )
    docs = text_splitter.split_documents(data)

    # Clean the document text
    cleaned_docs = [
        clean_text(doc.page_content if hasattr(doc, 'page_content') else doc) for doc in docs
    ]

    # Save the docs to a file for later use
    with open(docs_path, "wb") as f:
        pickle.dump(cleaned_docs, f)

    # Debugging: Check the structure of docs
    st.write(cleaned_docs[:3])  # Optional: Uncomment to inspect the structure of docs

    # Create FAISS vector store
    st.info("Creating embeddings... Please wait!")
    vectorstore = FAISS.from_texts(cleaned_docs, embedding_model)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    st.success("Data processed and embeddings saved!")

# Question Input
query = st.text_input("Ask a question about the content:")
if query and file_path and docs_path:
    try:
        # Load saved FAISS index and documents
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        # Retrieve top-k relevant chunks
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(query)

        # Build context from relevant chunks
        context = " ".join([doc for doc in relevant_docs])

        # Answer the query using QA pipeline
        result = qa_pipeline(question=query, context=context)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])
    except Exception as e:
        st.error(f"Error processing the query: {e}")
