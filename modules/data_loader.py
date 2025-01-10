import pandas as pd
from langchain.schema import Document


def load_data(file_path):
    """
    Load dataset from a CSV file and convert it into a list of Document objects.
    Each document represents a question-answer pair.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure the columns are present in the dataset
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            raise ValueError("CSV must have 'Question' and 'Answer' columns")

        # Convert each row to a Document object
        documents = []
        for _, row in df.iterrows():
            question = row['Question']
            answer = row['Answer']
            document = Document(page_content=f"Question: {question} Answer: {answer}")
            documents.append(document)
        
        return documents

    except Exception as e:
        print(f"Error in loading data: {e}")
        return []
