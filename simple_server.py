from typing import TypedDict
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchTextInput,
    BatchTextResponse,
    EnumParameterDescriptor,
    EnumVal,
    InputSchema,
    InputType,
    ParameterSchema,
    ResponseBody,
    TaskSchema,
    TextResponse,
)

server = MLServer(__name__)

server.add_app_metadata(
    name="Simple Server - Transform Case",
    author="Flask-ML Team",
    version="0.1.0",
    info=load_file_as_string("simple_server_info.md"),
)

class TransformCaseInputs(TypedDict):
    text_inputs: BatchTextInput


class TransformCaseParameters(TypedDict):
    to_case: str  # 'upper' or 'lower'


def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(key="text_inputs", label="Text to Transform", input_type=InputType.BATCHTEXT)
    parameter_schema = ParameterSchema(
        key="to_case",
        label="Case to Transform Text Into",
        subtitle="'upper' will convert all text to upper case. 'lower' will convert all text to lower case.",
        value=EnumParameterDescriptor(
            enum_vals=[EnumVal(key="upper", label="UPPER"), EnumVal(key="lower", label="LOWER")],
            default="upper",
        ),
    )
    return TaskSchema(inputs=[input_schema], parameters=[parameter_schema])


@server.route(
    "/transform_case",
    task_schema_func=create_transform_case_task_schema,
    short_title="Transform Case",
    order=0,
)
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    to_upper: bool = parameters["to_case"] == "upper"

    outputs = []
    for text_input in inputs["text_inputs"].texts:
        raw_text = text_input.text
        processed_text = raw_text.upper() if to_upper else raw_text.lower()
        outputs.append(TextResponse(value=processed_text, title=raw_text))

    return ResponseBody(root=BatchTextResponse(texts=outputs))


if __name__ == "__main__":
    # Run a debug server
    server.run()
