import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        # Convert the 'timestamp' column to Unix timestamps using the successful method
        if 'timestamp' in data.columns:
            data['timestamp'] = data['timestamp'].apply(lambda x: int(pd.to_datetime(x).timestamp()))
        return data
    except pd.errors.ParserError:
        print("Error: Issue parsing the CSV file.")
        return None

def split_data(data, test_size=0.2):
    return train_test_split(data, test_size=test_size)
