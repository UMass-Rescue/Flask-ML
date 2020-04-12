# Flask-ML
![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)

Flask-ML helps users easily deploy their ML models as a web service. Flask-ML Server, similar to Flask, allows the user to specify web services using the decorator pattern. But Flask-ML server allows users to specify float ndarray and other datatypes used in ML as input and output types of the function being decorated. 

Once the model is deployed using Flask-ML Server, users can then use Flask-ML client to call the web service made available through Flask-ML server. The client library provides a "predict" function that takes in the input to the ML webservice (float ndarray or other ML datatypes) and the endpoint of the service on the server as parameters, makes the call to the server and returns the output provided by the server (float ndarray or other ML datatypes). Flask-ML Client library is a library provided for convenience and isn't strictly required in order to make calls to Flask-ML Server. 

### Installation

To install Flask-ML server -
```
pip install flask-ml-server
```

To install Flask-ML client - 
```
pip install flask-ml-client
```

### Usage Example

In the following code, we will fit a Linear Regression model on the Boston Housing Price dataset (available in sklearn.datasets package) and make the housing price prediction model available as a web service.

#### Server Code - 

```Python3
from flask_ml_server import MLServer
from encoder_decoder import DTypes
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# loading the Boston Housing Price Prediction dataset
data = load_boston()
# fitting a Linear Regression model to the housing data
model = LinearRegression()
model.fit(data.data, data.target)

# make a server instance
serv = MLServer(__name__)

# adding the "housing_price_prediction" endpoint to the server. 
# input_type and output_type are required parameters. They should be one among the available types in encoder_decoder.DTypes
@serv.route('/housing_price_prediction', input_type=DTypes.FLOAT_NDARRAY, output_type=DTypes.FLOAT_NDARRAY)
def predict_housing_price(x):
    '''
    x :: np.ndarray - array of features
    returns :: np.ndarray - housing price prediction
    '''
    return model.predict(x)

# begin server instance
serv.run()
```

#### Client Code
```Python3
from flask_ml_client import MLClient
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
