#!/usr/bin/env python3
import unittest
import pandas as pd
import numpy as np
import tensorflow as tf

from preprocessing import data_loader, data_transformer
from neural_network import autoencoder
from anomaly_detection import detector, response

# Check for non-numeric values in the dataset
def check_for_non_numeric(data):
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Error: The following columns have non-numeric values: {', '.join(non_numeric_cols)}")
        return True
    return False

try:
    # Load data
    try:
        data = data_loader.load_data("data/theoretical_telemetry_data.csv")
    except FileNotFoundError:
        print("Error: Telemetry data file not found.")
        exit()

    # Check for non-numeric values
    if check_for_non_numeric(data):
        exit()
    
    # Normalize data
    try:
        normalized_data = data_transformer.normalize_data(data)
    except Exception as e:
        print(f"Error during data normalization: {e}")
        exit()
    
    # Build and train the autoencoder
    input_dim = normalized_data.shape[1]
    model = autoencoder.build_autoencoder(input_dim)
    X_train, X_test = data_loader.split_data(normalized_data)
    
    try:
        model.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))
    except Exception as e:
        print(f"Error during model training: {e}")
        exit()

    # Detect anomalies and respond in real-time (placeholder logic)
    new_data = X_test  # Placeholder for new incoming data
    try:
        anomalies = detector.detect_anomaly(model, new_data)
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        exit()
        
    if any(anomalies):
        try:
            response.trigger_response()
        except Exception as e:
            print(f"Error during response triggering: {e}")
            exit()

except Exception as e:
    print(f"Unexpected error: {e}")
