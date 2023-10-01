
import tensorflow as tf

def build_autoencoder(input_dim, encoding_dim=32):
    try:
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder
    except Exception as e:
        print(f"Error building the autoencoder: {e}")
        return None
