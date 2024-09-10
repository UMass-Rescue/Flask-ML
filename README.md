# Flask-ML

![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)

Flask-ML helps users easily deploy their ML models as a web service. Flask-ML Server, similar to Flask, allows the user to specify web services using the decorator pattern. But Flask-ML allows users to specify the input type being expected by the ML function and provides helper classes to form response objects for the outputs produced by the ML function.

### Installation

To install Flask-ML
```
pip install flask-ml
```

### Usage examples

#### Server

```
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.response import TextResponse


# Create a dummy ML model
class DummyModel:
    def __init__(self):
        self.counter = 0

    def predict(self, data: list) -> list:
        res = []
        for d in data:
            res.append({"counter": self.counter, "text": d["text"]})
            self.counter += 1
        return res


# create an instance of the model
model = DummyModel()

# Create a server
server = MLServer(__name__)


# Create an endpoint
@server.route("/dummymodel", DataTypes.TEXT)
def process_text(inputs: list, parameters: dict) -> dict:
    return TextResponse(model.predict(inputs)).get_response()


# Run the server (optional. You can also run the server using the command line)
server.run()
```

#### Client

```
from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

url = 'http://127.0.0.1:5000/dummymodel'  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = [
    {'text': 'Text to be classified'},
    {'text': 'Another text to be classified'}
]  # The inputs to be sent to the server
data_type = DataTypes.TEXT  # The type of the input data

response = client.request(inputs, data_type)  # Send a request to the server
print(response)  # Print the response
```
