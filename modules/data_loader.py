import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Loads and validates Q&A data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing questions and answers
        
    Returns:
        pandas.DataFrame: DataFrame containing the Q&A pairs
        
    Raises:
        Exception: If there's an error loading the data or if required columns are missing
    """
    try:
        data = pd.read_csv(file_path)
        if 'question' not in data.columns or 'answer' not in data.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns.")
        
        logger.info(f"Successfully loaded {len(data)} Q&A pairs from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise Exception(f"Error loading data: {e}")