from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import BatchFileInput, BatchTextInput, Input

DUMMY_MODEL_URL = "http://127.0.0.1:5000/dummymodel"  # The URL of the server
SENTIMENT_MODEL_URL = "http://127.0.0.1:5000/randomsentimentanalysis"
IMAGE_STYLE_TRANSFER_MODEL_URL = "http://127.0.0.1:5000/imagestyletransfer"


client = MLClient(DUMMY_MODEL_URL)  # Create an instance of the MLClient object


inputs = {
    "text_inputs": Input(
        root=BatchTextInput.model_validate(
            {"texts": [{"text": "Text to be classified"}, {"text": "Another text to be classified"}]}
        )
    )
}

parameters = {"model_parameter": 1.43}

response = client.request(inputs, parameters)  # Send a request to the server
print("dummy model response:")
print(response)  # Print the response
print()

parameters = {}

client.set_url(SENTIMENT_MODEL_URL)
response = client.request(inputs, parameters)
print("sentiment model response:")
print(response)
print()


client.set_url(IMAGE_STYLE_TRANSFER_MODEL_URL)
inputs = {
    "image_input": Input(
        root=BatchFileInput.model_validate(
            {"files": [{"path": "/path/to/image1.jpg"}, {"path": "/path/to/image2.jpg"}]}
        )
    )
}

response = client.request(inputs, parameters)
print("image style transfer model response:")
print(response)
print()
