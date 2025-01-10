
# Set environment variables

import os
import pandas as pd
import uuid



def load_data(data_path):
    """
    Loads data from a CSV file into a dictionary format.

    Args:
        data_path: Path to the CSV file.

    Returns:
        dict: A dictionary where keys are UUIDs and values are tuples of (question, answer).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist.")
    df = pd.read_csv(data_path)
    if df.empty or not all(col in df.columns for col in ['question', 'answer']):
        raise ValueError("Invalid dataset format")

    data_dict = {}
    for index, row in df.iterrows():
        data_dict[str(uuid.uuid4())] = (row['question'], row['answer']) 

    return data_dict

# Set environment variables 
# (This part is unclear, please provide specific environment variables to set)
# Example:
# os.environ["MY_API_KEY"] = "your_api_key"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_acc_key.json"