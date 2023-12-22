import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import flwr as fl   # Flower library for Federated Learning

# Define your columns based on features and target
features = ["身高", "體重", "BMI", "壓力感測1", "壓力感測2", "壓力感測3", "壓力感測4", "壓力感測5", "壓力感測6", "壓力感測7", "壓力感測8", "壓力感測9"]
target = "坐姿"
column_names = [target] + features  # Target is the first column

# Load your training and validation data
train_data_path = "client2_data/record_train2.csv"
valid_data_path = "client2_data/record_valid2.csv"

train_data = pd.read_csv(train_data_path, names=column_names, encoding='utf-8', index_col=False)
valid_data = pd.read_csv(valid_data_path, names=column_names, encoding='utf-8', index_col=False)

# Extract features and target labels
X_train = train_data[features].astype(float)
y_train = train_data[target]

X_valid = valid_data[features].astype(float)
y_valid = valid_data[target]

# Encode target labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)

# Define a simple neural network model using TensorFlow's Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Flower client setup
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    # def set_parameters(self, parameters):
    #     model.set_weights(parameters)

    def fit(self, parameters, config):  # Allow config to be optional
        model.set_weights(parameters)
        model.fit(X_train.values, y_train_encoded, epochs=1, batch_size=32)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):  # Allow config to be optional
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_valid.values, y_valid_encoded)
        return loss, len(X_valid.values), {"accuracy": accuracy}


client = MyClient()

# Create Flower client and start Federated Learning
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client) 

