from typing import TypedDict

import pytest

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *
from flask.wrappers import Response

from .constants import *


class MockResponse:
    def __init__(self, response: Response):
        self.status_code = response.status_code
        self.headers = {"Content-Type": "application/json"}
        self.response: Response = response

    def json(self):
        return self.response.get_json()


def create_response_model(results: BaseModel):
    return Response(response=results.model_dump_json(), status=200, mimetype="application/json")


def mock_post_request(url, json=None, **kwargs) -> MockResponse:
    data = RequestBody.model_validate(json)
    if url == "http://127.0.0.1:5000/process_text":
        return MockResponse(create_response_model(process_text(data.inputs["text_input"], data.parameters)))  # type: ignore
    elif url == "http://127.0.0.1:5000/process_file":
        return MockResponse(create_response_model(process_file(data.inputs["file_input"], data.parameters)))  # type: ignore
    elif url == "http://127.0.0.1:5000/process_texts":
        return MockResponse(create_response_model(process_texts(data.inputs["text_inputs"].root.texts, data.parameters)))  # type: ignore
    elif url == "http://127.0.0.1:5000/process_files":
        return MockResponse(create_response_model(process_files(data.inputs["file_inputs"].root.files, data.parameters)))  # type: ignore
    assert False, "Never"


def process_text(inputs: TextInput, parameters):
    results = TextResponse(title=inputs.text, value="processed_text.txt")
    return results


def process_file(inputs: FileInput, parameters):
    results = FileResponse(title=inputs.path, path="processed_image.img", file_type=FileType.IMG)
    return results


def process_texts(inputs: List[TextInput], parameters):
    results = [TextResponse(title=inp.text, value="processed_text.txt") for inp in inputs]
    results = BatchTextResponse(texts=results)
    return results


def process_files(inputs: List[FileInput], parameters):
    results = [
        FileResponse(title=inp.path, path="processed_image.img", file_type=FileType.IMG) for inp in inputs
    ]
    results = BatchFileResponse(files=results)
    return results


def process_directory(inputs: DirectoryInput, parameters):
    results = DirectoryResponse(title=inputs.path, path="processed_directory")
    return results


def process_directories(inputs: List[DirectoryInput], parameters):
    results = [DirectoryResponse(title=inp.path, path="processed_directory") for inp in inputs]
    results = BatchDirectoryResponse(directories=results)
    return results


@pytest.fixture(scope="session", autouse=True)
def server():
    server = MLServer(__name__)

    class SingleTextInput(TypedDict):
        text_input: TextInput

    class TextInputs(TypedDict):
        text_inputs: BatchTextInput

    class SingleFileInput(TypedDict):
        file_input: FileInput

    class FileInputs(TypedDict):
        file_inputs: BatchFileInput

    class SingleDirectoryInput(TypedDict):
        dir_input: DirectoryInput

    class DirectoryInputs(TypedDict):
        dir_inputs: BatchDirectoryInput

    class TextParameters(TypedDict):
        param1: str

    class IntParameters(TypedDict):
        param1: int

    class FloatParameters(TypedDict):
        param1: float

    class EnumParameters(TypedDict):
        param1: str

    @server.route("/process_text")
    def server_process_text(inputs: SingleTextInput, parameters: TextParameters) -> ResponseBody:
        return ResponseBody(root=process_text(inputs["text_input"], parameters))

    @server.route("/process_texts")
    def server_process_texts(inputs: TextInputs, parameters: TextParameters) -> ResponseBody:
        return ResponseBody(root=process_texts(inputs["text_inputs"].texts, parameters))

    @server.route("/process_file")
    def server_process_image(inputs: SingleFileInput, parameters: FloatParameters) -> ResponseBody:
        return ResponseBody(root=process_file(inputs["file_input"], parameters))

    @server.route("/process_files")
    def server_process_images(inputs: FileInputs, parameters: FloatParameters) -> ResponseBody:
        return ResponseBody(root=process_files(inputs["file_inputs"].files, parameters))

    @server.route("/process_invalid")
    def server_process_invalid(inputs: FileInputs, parameters: FloatParameters) -> ResponseBody:
        raise Exception("Internal Server Error")

    def get_task_schema(inputSchema: InputSchema, parameterSchema: ParameterSchema):
        return lambda: TaskSchema(
            inputs=[inputSchema],
            parameters=[parameterSchema],
        )

    @server.route("/process_text_with_schema", get_task_schema(TEXT_INPUT_SCHEMA, TEXT_PARAM_SCHEMA))
    def server_process_text_with_schema(inputs: SingleTextInput, parameters: TextParameters) -> ResponseBody:
        return ResponseBody(root=process_text(inputs["text_input"], parameters))

    @server.route("/process_texts_with_schema", get_task_schema(BATCHTEXT_INPUT_SCHEMA, INT_PARAM_SCHEMA))
    def server_process_texts_with_schema(inputs: TextInputs, parameters: IntParameters) -> ResponseBody:
        return ResponseBody(root=process_texts(inputs["text_inputs"].texts, parameters))

    @server.route("/process_file_with_schema", get_task_schema(FILE_INPUT_SCHEMA, FLOAT_PARAM_SCHEMA))
    def server_process_image_with_schema(
        inputs: SingleFileInput, parameters: FloatParameters
    ) -> ResponseBody:
        return ResponseBody(root=process_file(inputs["file_input"], parameters))

    @server.route(
        "/process_files_with_schema", get_task_schema(BATCHFILE_INPUT_SCHEMA, RANGED_FLOAT_PARAM_SCHEMA)
    )
    def server_process_images_with_schema(inputs: FileInputs, parameters: FloatParameters) -> ResponseBody:
        return ResponseBody(root=process_files(inputs["file_inputs"].files, parameters))

    @server.route(
        "/process_invalid_with_schema", get_task_schema(BATCHFILE_INPUT_SCHEMA, RANGED_FLOAT_PARAM_SCHEMA)
    )
    def server_process_invalid_with_schema(inputs: FileInputs, parameters: FloatParameters) -> ResponseBody:
        raise Exception("Internal Server Error")

    @server.route(
        "/process_directory_and_enum_parameter_with_schema",
        get_task_schema(DIRECTORY_INPUT_SCHEMA, ENUM_PARAM_SCHEMA),
    )
    def server_process_directory_and_enum_parameter_with_schema(
        inputs: SingleDirectoryInput, parameters: EnumParameters
    ) -> ResponseBody:
        return ResponseBody(root=process_directory(inputs["dir_input"], parameters))

    @server.route(
        "/process_directories_and_ranged_int_parameter_with_schema",
        get_task_schema(BATCHDIRECTORY_INPUT_SCHEMA, RANGED_INT_PARAM_SCHEMA),
    )
    def server_process_directories_and_ranged_int_parameter_with_schema(
        inputs: DirectoryInputs, parameters: IntParameters
    ) -> ResponseBody:
        return ResponseBody(root=process_directories(inputs["dir_inputs"].directories, parameters))
    
    @server.route(
        "/process_text_input_with_text_area_schema",
        get_task_schema(TEXTAREA_INPUT_SCHEMA, TEXT_PARAM_SCHEMA),
    )
    def server_process_text_input_with_text_area_schema(
        inputs: SingleTextInput, parameters: TextParameters
    ) -> ResponseBody:
        return ResponseBody(root=process_text(inputs["text_input"], parameters))

    return server


@pytest.fixture(scope="session", autouse=True)
def app(server: MLServer):
    return server.app.test_client()
