from typing import TypedDict

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    BatchTextInput,
    BatchTextResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
    TextInput,
    TextResponse,
)


class SentimentModel:
    def predict(self, data: list[TextInput]) -> list[dict[str, str]]:
        return [
            {"text": t.text, "sentiment": "positive" if i % 2 == 0 else "negative"}
            for i, t in enumerate(data)
        ]


# create an instance of the model
sentiment_model = SentimentModel()

# Create a server
server = MLServer(__name__)

SENTIMENT_ANALYSIS_TASK_SCHEMA = TaskSchema(
    inputs=[
        InputSchema(key="text_inputs", label="Choose a set of text inputs", input_type=InputType.BATCHTEXT)
    ],
    parameters=[],
)


def create_task_schema() -> TaskSchema:
    return SENTIMENT_ANALYSIS_TASK_SCHEMA


class SentimentInputs(TypedDict):
    text_inputs: BatchTextInput


class Parameters(TypedDict): ...


# You can try adding input_schema = SENTIMENT_ANALYSIS_INPUT_SCHEMA below to the route decorator, and see how calling /api/routes changes
@server.route("/randomsentimentanalysis", task_schema_func=create_task_schema)
def sentiment_analysis(inputs: SentimentInputs, parameters: Parameters) -> ResponseBody:
    results = sentiment_model.predict(inputs["text_inputs"].texts)
    text_results = [TextResponse(value=res["sentiment"]) for res in results]
    response = BatchTextResponse(texts=text_results)
    return ResponseBody(root=response)


# Run the server (optional. You can also run the server using the command line)
server.run()
