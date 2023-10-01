from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    try:
        # Exclude 'timestamp' column during normalization
        columns_to_normalize = data.columns.difference(['timestamp'])
        scaler = MinMaxScaler()
        data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
        return data
    except ValueError:
        print("Error: Issue during data normalization. Ensure data is numeric.")
        return None
