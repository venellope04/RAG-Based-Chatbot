# modules/rag_pipeline.py
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Initialize logging
logging.basicConfig(level=logging.INFO)

class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __init__(self):
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def initialize_system(data_dict):
    try:
        logging.info("Initializing system...")

        # Initialize embedding model
        embed_model = CustomHuggingFaceEmbeddings()
        logging.info("Embedding model initialized")

        # Create documents
        documents = []
        for doc_id, (question, answer) in data_dict.items():
            text = f"Question: {question}\nAnswer: {answer}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={"id": doc_id, "question": question, "answer": answer}
                )
            )

        logging.info(f"Created {len(documents)} documents")

        # Create FAISS vectorstore
        try:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embed_model
            )
            logging.info("FAISS vectorstore created successfully")
        except Exception as e:
            logging.error(f"Failed to create vectorstore: {e}")
            raise

      

        # Initialize LLM
        try:
            llm = HuggingFaceHub(
                repo_id="tiiuae/falcon-7b-instruct",
                model_kwargs={
                    "temperature": 0.5,
                    "max_length": 512,
                    "do_sample": True
                }
            )
            logging.info("LLM initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise

       

        except Exception as e:
            logging.error(f"Failed to create RAG chain: {e}")
            raise

    except Exception as e:
        logging.error(f"System initialization failed: {e}")
        raise


            
    except Exception as e:
        logging.error(f"Error getting response: {e}")
        raise