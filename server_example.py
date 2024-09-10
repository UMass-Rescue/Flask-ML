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

# Expected request json format:
# {
#     "inputs": [
#         {"text": "Text to be classified"},
#         {"text": "Another text to be classified"}
#     ],
#     "data_type": "TEXT",
#     "parameters": {}
# }
