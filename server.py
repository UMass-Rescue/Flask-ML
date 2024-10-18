from turtle import title
from typing import List, TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    BatchTextInput,
    InputSchema,
    InputType,
    ParameterSchema,
    ResponseBody,
    BatchTextResponse,
    TaskSchema,
    TextResponse,
    EnumVal,
)
from flask_ml.flask_ml_server.utils import EnumParameterDescriptor

server = MLServer(__name__)


class TranscribeTaskInputs(TypedDict):
    text_inputs: BatchTextInput


class TranscribeTaskParameters(TypedDict):
    transform_case: str  # 'upper' or 'lower'


def create_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
                key="text_inputs",
                label="A collection of text inputs",
                input_type=InputType.BATCHTEXT,
            )
        ],
        parameters=[
            ParameterSchema(
                key="transform_case",
                label="Transform case action",
                subtitle="UPPER will convert all text to upper case, LOWER will convert all text to lower case",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(
                            key="upper",
                            label="UPPER",
                        ),
                        EnumVal(
                            key="lower",
                            label="LOWER",
                        ),
                    ],
                    default="upper",
                ),
            )
        ],
    )


@server.route("/transform_case", task_schema_func=create_task_schema, short_title="Transform Case", order=0)
def transform_case(inputs: TranscribeTaskInputs, parameters: TranscribeTaskParameters) -> ResponseBody:
    to_upper: bool = parameters["transform_case"] == "upper"

    outputs: List[TextResponse] = []
    for text_input in inputs["text_inputs"].texts:
        raw_text = text_input.text
        processed_text = raw_text.upper() if to_upper else raw_text.lower()
        outputs.append(
            TextResponse(value=processed_text, title=raw_text)
        )  # raw_text is a link to original text

    return ResponseBody(root=BatchTextResponse(texts=outputs))


# Run a debug flask server
server.run()
