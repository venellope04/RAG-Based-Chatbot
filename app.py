import streamlit as st
import uuid
import logging
from modules.data_loader import load_data
from modules.rag_pipeline import initialize_system
from modules.dialogFlow_handler import DialogflowHandler
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
DIALOGFLOW_PROJECT_ID = 'inductive-time-447222-k2'
DIALOGFLOW_LANGUAGE_CODE = 'en'

# Global variables
rag_chain = None
dialogflow_handler = None

def initialize_dialogflow():
    """Initialize Dialogflow handler with error checking"""
    try:
        global dialogflow_handler
        dialogflow_handler = DialogflowHandler(
            project_id=DIALOGFLOW_PROJECT_ID,
            language_code=DIALOGFLOW_LANGUAGE_CODE
        )
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Dialogflow: {e}")
        return False

def generate_response(prompt):
    """
    Generates a response using both RAG and Dialogflow
    """
    try:
        logging.info(f"Processing prompt: {prompt}")

        if not rag_chain:
            logging.error("System not initialized")
            return "System is initializing. Please try again later."

        # Get chat history
        chat_history = st.session_state.get("chat_history", [])

        # Get RAG response
        rag_response = rag_chain.run(
            input_data={
                "question": prompt,
                "chat_history": chat_history
            }
        ) 

        if not isinstance(rag_response, dict) or 'answer' not in rag_response:
            rag_text = "I couldn't find a relevant answer in my knowledge base."
        else:
            rag_text = rag_response['answer']

        # Get Dialogflow response if available
        dialogflow_text = None
        if dialogflow_handler:
            try:
                dialogflow_text, intent_name, confidence = dialogflow_handler.detect_intent(
                    text=prompt
                )  # Use 'text' directly
                logging.info(f"Dialogflow intent: {intent_name} (confidence: {confidence})")
            except Exception as e:
                logging.error(f"Dialogflow error: {e}")
                dialogflow_text = None

        # Construct final response
        if dialogflow_text:
            return f"Dialogflow: {dialogflow_text}\n\nKnowledge Base: {rag_text}"
        return f"Knowledge Base: {rag_text}"

    except Exception as e:
        logging.error(f"Error in generate_response: {e}")
        return f"Error: {str(e)}"

def main():
    st.title("AI Knowledge Assistant")
    st.write("Hello! Ask me anything.")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # System initialization
    try:
        # Initialize RAG system
        data_dict = load_data('data.csv') 
        global rag_chain
        rag_chain = initialize_system(data_dict) 

        # Initialize Dialogflow
        if initialize_dialogflow():
            st.success("System initialized successfully (RAG + Dialogflow)")
        else:
            st.warning("System initialized with RAG only (Dialogflow initialization failed)")

    except Exception as e:
        st.error(f"System initialization error: {e}")
        return

    # User interface
    user_input = st.text_input("You:")
    if user_input:
        response = generate_response(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

if __name__ == "__main__":
    main()