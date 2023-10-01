
import tensorflow as tf

def detect_anomaly(model, new_data, threshold=0.05):
    try:
        reconstructed_data = model.predict(new_data)
        mse = tf.keras.losses.MSE(new_data, reconstructed_data)
        anomalies = mse > threshold
        return anomalies.numpy()
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        return None
