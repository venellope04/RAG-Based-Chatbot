import streamlit as st
import logging
from modules.data_loader import load_data
from modules.rag_pipeline import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_PATH = 'data.csv'

def initialize_session_state():
    """
    Initializes the RAG system and stores it in session state.
    """
    try:
        data = load_data(DATA_PATH)
        st.session_state.rag_system = RAGSystem(data)
        st.success("Knowledge base initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        logger.error(f"Initialization error: {e}")

def main():
    """
    Main application function that sets up the Streamlit interface.
    """
    st.title("AI Knowledge Assistant")
    st.write("Hello! Ask me anything about our knowledge base.")

    if 'rag_system' not in st.session_state:
        initialize_session_state()

    user_input = st.text_input("You:")
    
    if user_input:
        try:
            response = st.session_state.rag_system.generate_response(user_input)
            st.write(f"**Assistant:** {response}")
        except Exception as e:
            st.error(f"Error processing your question: {e}")
            logger.error(f"Error in response generation: {e}")

if __name__ == "__main__":
    main()