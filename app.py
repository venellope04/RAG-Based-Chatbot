import streamlit as st
import uuid
import logging
from modules.data_loader import load_data
from modules.rag_pipeline import initialize_system
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Global variables
rag_chain = None


def generate_response(user_input, rag_chain):
    """
    Generates a response using the RAG pipeline, providing only the relevant answer to the current question.
    """
    try:
        logging.info(f"Processing prompt: {user_input, rag_chain}")

        if not rag_chain:
            logging.error("System not initialized")
            return "System is initializing. Please try again later."

        # Structure the input for the RAG model with only the current question
        input_data = {
            "question": user_input,
            "chat_history": []  # No history, just the current question
        }

        # Get RAG response using the structured input
        rag_response = rag_chain.invoke(input_data)

        # If the response is not in the expected format or doesn't contain an answer, return a default message
        if not isinstance(rag_response, dict) or 'answer' not in rag_response:
            rag_text = "I couldn't find a relevant answer in my knowledge base."
        else:
            rag_text = rag_response['answer']

        return rag_text

    except Exception as e:
        logging.error(f"Error in generate_response: {e}")
        return f"Error: {str(e)}"




def main():
    st.title("AI Knowledge Assistant")
    st.write("Hello! Ask me anything.")

    # Initialize system (ensure correct file path to your dataset)
    try:
        rag_chain = initialize_system("data.csv")  # Pass the correct path to your dataset
        st.success("System initialized successfully")

    except Exception as e:
        st.error(f"System initialization error: {e}")
        return

    # User interface for chat interaction
    user_input = st.text_input("You:")
    if user_input:
        # You can implement a method to generate responses here (e.g., using rag_chain)
        response = generate_response(user_input, rag_chain)  # Use rag_chain to get response
        st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()

