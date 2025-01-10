from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from modules.data_loader import load_data
import logging
import os


def initialize_system(file_path):
    """
    Initialize the RAG system with the dataset from the file path.
    """
    try:
        logging.info("Initializing system...")

        # Load the dataset as Document objects
        documents = load_data(file_path)
        if not documents:
            raise ValueError("No documents were loaded.")

        # Load pre-trained model for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create the FAISS vectorstore
        vectorstore = FAISS.from_documents(documents, embeddings)
        logging.info("FAISS vectorstore created successfully")

        # Initialize the LLM (Language Model) for generating responses
        # Specify the model repository ID in the 'repo_id'
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Replace with the actual model ID you want to use
            huggingfacehub_api_token = "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",
            model_kwargs={"do_sample": True}
        )

        # Setup the retriever using the vectorstore
        retriever = vectorstore.as_retriever()

        # Set up the ConversationalRetrievalChain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            return_source_documents=False  # Only the answer, no source documents
        )

        logging.info("LLM initialized successfully")

        return rag_chain

    except Exception as e:
        logging.error(f"System initialization error: {e}")
        raise
