from typing import List, TypedDict

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    BatchFileResponse,
    BatchTextInput,
    BatchTextResponse,
    FileInput,
    FileResponse,
    FileType,
    FloatRangeDescriptor,
    InputSchema,
    InputType,
    ParameterSchema,
    RangedFloatParameterDescriptor,
    ResponseBody,
    TaskSchema,
    TextInput,
    TextResponse,
)


# Create a dummy ML model
class DummyModel:
    def predict(self, data: list) -> list[str]:
        return [str(e) for e in range(len(data))]  # Return 0 to len(data) - 1


class SentimentModel:
    def predict(self, data: list[TextInput]) -> list[dict[str, str]]:
        return [
            {"text": t.text, "sentiment": "positive" if i % 2 == 0 else "negative"}
            for i, t in enumerate(data)
        ]


class ImageStyleTransferModel:
    def predict(self, data: list[FileInput]) -> list[dict[str, str]]:
        return [{"result": f"stylized_image_{i}.jpg"} for i, f in enumerate(data)]


# create an instance of the model
model = DummyModel()
sentiment_model = SentimentModel()
image_style_transfer_model = ImageStyleTransferModel()

# Create a server
server = MLServer(__name__)


PROCESS_TEXT_INPUT_SCHEMA = TaskSchema(
    inputs=[
        InputSchema(
            key="text_input_list", label="Choose several text inputs", input_type=InputType.BATCHTEXT
        ),
        InputSchema(key="db_name", label="Database name", input_type=InputType.TEXT),
    ],
    parameters=[
        ParameterSchema(
            key="model_parameter",
            label="Model parameter",
            value=RangedFloatParameterDescriptor(range=FloatRangeDescriptor(min=0, max=1), default=0.5),
        )
    ],
)


class TextInputs(TypedDict):
    text_input_list: BatchTextInput
    db_name: TextInput


class TextParameters(TypedDict):
    model_parameter: float


# Create an endpoint
@server.route("/tasks/generate_db", input_schema=PROCESS_TEXT_INPUT_SCHEMA)
def process_text(inputs: TextInputs, parameters: TextParameters) -> ResponseBody:
    # Inputs
    batch_of_text: BatchTextInput = inputs["text_input_list"]
    list_of_texts: List[TextInput] = batch_of_text.texts
    db_name = inputs["db_name"].text

    # Parameters
    float_param_value = parameters["model_parameter"]

    print(list_of_texts[0].text)
    print(db_name)
    print(float_param_value)

    predictions = model.predict([txtModel.text.capitalize() for txtModel in list_of_texts])

    result_texts = [TextResponse(value=p) for p in predictions]
    response = BatchTextResponse(texts=result_texts)
    return ResponseBody(root=response)


SENTIMENT_ANALYSIS_INPUT_SCHEMA = TaskSchema(
    inputs=[
        InputSchema(key="text_input", label="Choose a set of text inputs", input_type=InputType.BATCHTEXT)
    ],
    parameters=[],
)


class SentimentInputs(TypedDict):
    text_inputs: BatchTextInput


@server.route("/randomsentimentanalysis", input_schema=SENTIMENT_ANALYSIS_INPUT_SCHEMA)
def sentiment_analysis(inputs: SentimentInputs, parameters: dict) -> ResponseBody:
    results = sentiment_model.predict(inputs["text_inputs"].texts)
    text_results = [TextResponse(value=res["sentiment"]) for res in results]
    response = BatchTextResponse(texts=text_results)
    return ResponseBody(root=response)


IMAGE_STYLE_TRANSFER_SCHEMA = TaskSchema(
    inputs=[
        InputSchema(key="image_input", label="Choose a set of image inputs", input_type=InputType.BATCHFILE)
    ],
    parameters=[],
)


class ImageInput(TypedDict):
    image_input: BatchFileInput


@server.route("/imagestyletransfer", input_schema=IMAGE_STYLE_TRANSFER_SCHEMA)
def image_style_transfer(inputs: ImageInput, parameters: dict) -> ResponseBody:
    results = image_style_transfer_model.predict(inputs["image_input"].files)
    image_results = [FileResponse(file_type=FileType.IMG, path=res["result"]) for res in results]
    response = BatchFileResponse(files=image_results)
    return ResponseBody(root=response)


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
