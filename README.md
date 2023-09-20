
# AthenaDetect: AI-Enabled Satellite Cybersecurity

`AthenaDetect` is an advanced AI model designed for anomaly detection in satellite telemetry data. Inspired by the wisdom of Athena, the Greek goddess, this program seeks to protect satellite systems by identifying any unusual patterns in the incoming data stream, signaling potential security breaches or system malfunctions.

## Requirements

- **Python 3.8+**: Ensure you have a recent version of Python installed.
- **Python Packages**: Install the necessary Python packages using the following:
    ```
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```

## Features

1. **Data Preprocessing**: Handles raw telemetry data and preprocesses it for neural network training.
2. **Autoencoder Neural Network**: Utilizes a deep learning approach to learn the normal patterns of satellite telemetry data and detects anomalies based on reconstruction error.
3. **Visualization**: Plots the reconstruction error to help in threshold determination for anomaly detection.

## Directory Structure

- `data/`: Contains the satellite telemetry data.
- `models/`: Where trained models will be saved.
- `data_loader.py`: Handles data loading and preprocessing.
- `model.py`: Contains the neural network model, training, and evaluation.
- `main.py`: The main script to run the program.

## How to Use

1. **Data Preparation**: Place your satellite telemetry data in the `data/` directory. The expected format is a CSV file.
2. **Train the Model**: Run the `main.py` script to preprocess the data and train the autoencoder neural network:
    ```
    python main.py
    ```
3. **Anomaly Detection**: Once trained, the model will automatically detect anomalies in the data and flag them.
4. **Visualization**: Examine the generated plots to understand the reconstruction error distribution and set the appropriate threshold for anomaly detection.

## Future Enhancements

- Integration with real-time satellite telemetry streams.
- Incorporation of additional satellite communication protocol data for richer feature extraction.
