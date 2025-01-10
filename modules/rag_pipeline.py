from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, data):
        """
        Initializes the RAG system with the provided dataset.
        
        Args:
            data (pandas.DataFrame): DataFrame containing questions and answers
        """
        self.data = data
        self.knowledge_base = None
        self.initialize_system()

    def create_documents(self):
        """
        Creates document objects from the dataset.
        
        Returns:
            list: List of Document objects containing questions and their corresponding answers
        """
        documents = []
        for _, row in self.data.iterrows():
            doc = Document(
                page_content=row['question'],
                metadata={'answer': row['answer']}
            )
            documents.append(doc)
        return documents

    def initialize_system(self):
        """
        Sets up the vectorstore and embedding system using document format.
        """
        try:
            embedding_function = SentenceTransformerEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            
            documents = self.create_documents()
            
            self.knowledge_base = FAISS.from_documents(
                documents=documents,
                embedding=embedding_function
            )
            
            logger.info(f"RAG system initialized successfully with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def generate_response(self, query, similarity_threshold=1.5):
        """
        Generates a response for the given query using document-based search.
        
        Args:
            query (str): The user's question
            similarity_threshold (float): Maximum similarity score for considering an answer relevant
            
        Returns:
            str: The most relevant answer from the knowledge base
        """
        try:
            search_results = self.knowledge_base.similarity_search_with_score(query, k=1)
            
            if not search_results:
                return "No relevant answer found in the knowledge base."
                
            best_match, score = search_results[0]
            
            if score > similarity_threshold:
                return "I couldn't find a sufficiently relevant answer in the knowledge base."
                
            return best_match.metadata['answer']
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"An error occurred while processing your question: {str(e)}"