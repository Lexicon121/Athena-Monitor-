
import tensorflow as tf

def detect_anomaly(model, new_data, threshold=0.05):
    reconstructed_data = model.predict(new_data)
    mse = tf.keras.losses.MSE(new_data, reconstructed_data)
    anomalies = mse > threshold
    return anomalies.numpy()
