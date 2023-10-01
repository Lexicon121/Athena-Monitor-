
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError:
        print("Error: Issue parsing the CSV file.")
        return None

def split_data(data, test_size=0.2):
    return train_test_split(data, test_size=test_size)
