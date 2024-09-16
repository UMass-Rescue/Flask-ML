from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import (ImageResult, ResponseModel,
                                             TextResult)


# Create a dummy ML model
class DummyModel:
    def predict(self, data: list) -> list:
        return [str(e) for e in range(len(data))]  # Return 0 to len(data) - 1


class SentimentModel:
    def predict(self, data: list[str]) -> list[str]:
        return [
            {"text": t.text, "sentiment": "positive" if i % 2 == 0 else "negative"}
            for i, t in enumerate(data)
        ]


class ImageStyleTransferModel:
    def predict(self, data: list[str]) -> list[str]:
        return [
            {"file_path": f.file_path, "result": f"stylized_image_{i}.jpg"}
            for i, f in enumerate(data)
        ]


# create an instance of the model
model = DummyModel()
sentiment_model = SentimentModel()
image_style_transfer_model = ImageStyleTransferModel()

# Create a server
server = MLServer(__name__)


# Create an endpoint
@server.route("/dummymodel", DataTypes.TEXT)
def process_text(inputs: list, parameters: dict) -> dict:
    results = model.predict(inputs)
    results = [TextResult(text=e.text, result=r) for e, r in zip(inputs, results)]
    response = ResponseModel(results=results)
    return response.get_response()


@server.route("/randomsentimentanalysis", DataTypes.TEXT)
def sentiment_analysis(inputs: list, parameters: dict) -> dict:
    results = sentiment_model.predict(inputs)
    results = [TextResult(text=res["text"], result=res["sentiment"]) for res in results]
    response = ResponseModel(results=results)
    return response.get_response()


@server.route("/imagestyletransfer", DataTypes.IMAGE)
def image_style_transfer(inputs: list, parameters: dict) -> dict:
    results = image_style_transfer_model.predict(inputs)
    results = [
        ImageResult(file_path=res["file_path"], result=res["result"]) for res in results
    ]
    response = ResponseModel(results=results)
    return response.get_response()


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
