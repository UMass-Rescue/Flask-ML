from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

DUMMY_MODEL_URL = "http://127.0.0.1:5000/dummymodel"  # The URL of the server
SENTIMENT_MODEL_URL = "http://127.0.0.1:5000/randomsentimentanalysis"
IMAGE_STYLE_TRANSFER_MODEL_URL = "http://127.0.0.1:5000/imagestyletransfer"


client = MLClient(DUMMY_MODEL_URL)  # Create an instance of the MLClient object

inputs = [
    {"text": "Text to be classified"},
    {"text": "Another text to be classified"},
]  # The inputs to be sent to the server
data_type = DataTypes.TEXT  # The type of the input data

response = client.request(inputs, data_type)  # Send a request to the server
print("dummy model response:")
print(response)  # Print the response


client.set_url(SENTIMENT_MODEL_URL)
response = client.request(inputs, data_type)
print("sentiment model response:")
print(response)


client.set_url(IMAGE_STYLE_TRANSFER_MODEL_URL)
data_type = DataTypes.IMAGE
inputs = [{"file_path": "/path/to/image1.jpg"}, {"file_path": "/path/to/image2.jpg"}]
response = client.request(inputs, data_type)
print("image style transfer model response:")
print(response)
