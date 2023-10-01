#!/usr/bin/env python3

from preprocessing import data_loader, data_transformer
from neural_network import autoencoder
from anomaly_detection import detector, response

try:
    # Load and preprocess data
    try:
        data = data_loader.load_data("data/satellite_telemetry_data.csv")
    except FileNotFoundError:
        print("Error: Telemetry data file not found.")
        exit()
    
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
