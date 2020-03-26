# Flask-ML
## MLClient

The MLClient object is a wrapper for making requests to an instance of the MLServer. It provides useful services for a client to make prediction calls to the server and receive outputs without having to format data

### Installation

```
pip install flask-ml-client
```

### Usage
```Python3
from flask_ml_client.ml_client import MLClient
from sklearn.datasets import load_boston

data = load_boston()
datapoint_idx = 2

# the address at which the server is running.
# 127.0.0.1 is the IP Address. 5000 is the port number. 
# When "MLServer" is run, it'll output this address next to "Running on " text.
HOST = 'http://127.0.0.1:5000'

client = MLClient(HOST)

# get_models() returns all the available endpoints on the specified HOST
models = client.get_models()
print("Models: {}".format(models))

# x is the input vector of features for which we need to predict the housing price
x = data.data[datapoint_idx][None]
print("Input data type : {}".format(type(x)))
print("Input Data shape : {}".format(x.shape))

# the endpoint we want to use in the server
endpoint = 'housing_price_prediction'
# sending the data to the endpoint on host and getting the results back
result = client.predict(x, endpoint)

print("Result data type : {}".format(type(result)))
print(f'Actual value of the house : {data.target[datapoint_idx]}')
print(f'Predicted value of the house : {result}')
```
