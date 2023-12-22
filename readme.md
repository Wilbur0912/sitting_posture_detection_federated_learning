# Federated Learning Note
---
## How it operates?
![image](https://github.com/Wilbur0912/sitting_posture_detection_federated_learning/assets/89004015/e3da12fe-f692-49ca-b946-421479a3b9f9)



---
## Client1.py
Note: I created 2 clients, each client has different data

import dependencies
```python=
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import flwr as fl   # Flower library for Federated Learning
```

Define columns based on features and target
```python=
# Define your columns based on features and target
features = ["height", "weight", "BMI", "FSR1", "FSR2", "FSR3", "FSR4", "FSR5", "FSR6", "FSR7", "FSR8", "FSR9"]
target = "posture"
column_names = [target] + features  # Target is the first column
```
### Load training and validation data
```python=
train_data_path = "client1_data/record_train1.csv"
valid_data_path = "client1_data/record_valid1.csv"

train_data = pd.read_csv(train_data_path, names=column_names, encoding='utf-8', index_col=False)
valid_data = pd.read_csv(valid_data_path, names=column_names, encoding='utf-8', index_col=False)
```

### Extract features and target data
```python=
X_train = train_data[features].astype(float)
y_train = train_data[target]

X_valid = valid_data[features].astype(float)
y_valid = valid_data[target]
```
### Encode target labels using LabelEncoder
```python=
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)
```
### Define a simple neural network model
```python=
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])
```
### Compile the model
```python=
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
### Flower client setup
```python=
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):  
        model.set_weights(parameters)
        model.fit(X_train.values, y_train_encoded, epochs=1, batch_size=32)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config): 
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_valid.values, y_valid_encoded)
        return loss, len(X_valid.values), {"accuracy": accuracy}
    
client = MyClient()
```
### Create Flower client and start Federated Learning
```python=
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client) 
```
---
## Server.py
### import dependencies
```python =
import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
```

### create strategy
```python=
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)
```

### start server
```python=
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=10),
    strategy = strategy
)
```
---
## Test Result
![image](https://github.com/Wilbur0912/sitting_posture_detection_federated_learning/assets/89004015/dced0b1e-2422-453a-a557-90e17101295d)
### losses
```
app_fit: losses_distributed [(1, 2.003454980511426), (2, 1.541882242521938), (3, 1.3554413757951285), (4, 1.4613317687699414), (5, 1.4867502408519495), (6, 1.481644267558903), (7, 1.2023357226344127), (8, 1.3388771874765697), (9, 1.3960736465290648), (10, 0.928022098325799)]
```
### accuracy
```
app_fit: metrics_distributed {'accuracy': [(1, 0.6932377683481126), (2, 0.7965098212574382), (3, 0.8430040397239869), (4, 0.8598317448355097), (5, 0.8690557571094926), (6, 0.8761607989450599), (7, 0.8757868591741299), (8, 0.8833904749584108), (9, 0.8839513846148059), (10, 0.8803988823158119)]}
```
