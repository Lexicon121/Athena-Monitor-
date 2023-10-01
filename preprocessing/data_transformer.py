
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    try:
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    except ValueError:
        print("Error: Issue during data normalization. Ensure data is numeric.")
        return None
