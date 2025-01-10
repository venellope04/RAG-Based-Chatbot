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

def set_page_config():
    """
    Configures the Streamlit page settings and styling.
    """
    st.set_page_config(
        page_title="AI Knowledge Assistant",
        page_icon="🤖",
        layout="centered"
    )

    # Custom CSS styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            padding: 15px;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            background-color: #f0f2f6;
        }
        .assistant-message {
            background-color: #e8f0fe;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """
    Initializes the RAG system and stores it in session state.
    """
    with st.spinner("Initializing the knowledge base..."):
        try:
            data = load_data(DATA_PATH)
            st.session_state.rag_system = RAGSystem(data)
            st.session_state.chat_history = []
            st.success("✨ Knowledge base initialized successfully")
            st.success("This bot utilizes a massive dataset of human conversations to generate responses. Feel free to engage in a conversation with it. Let's chat. 💬")
        except Exception as e:
            st.error(f"❌Failed to initialize system: {e}")
            logger.error(f"Initialization error: {e}")

def main():
    """
    Main application function that sets up the Streamlit interface.
    """
    set_page_config()
    
    st.title("🤖 AI Knowledge Assistant")
    st.markdown("---")


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