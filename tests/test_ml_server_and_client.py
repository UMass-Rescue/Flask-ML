from typing import List, TypedDict
from unittest.mock import patch

from flask.json import jsonify
from flask.wrappers import Response
import pytest

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *

from .constants import *


class MockResponse:
    def __init__(self, response):
        self.status_code = response.status_code
        self.headers = {"Content-Type": "application/json"}
        self.response: Response = response

    def json(self):
        return self.response.get_json()


def create_response_model(results: BaseModel):
    return Response(response=results.model_dump_json(), status=200, mimetype="application/json")


def mock_post_request(url, json=None, **kwargs):
    data = RequestBody.model_validate(json)
    if url == "http://127.0.0.1:5000/process_text":
        return MockResponse(create_response_model(process_text(data.inputs["text_inputs"].root.texts, data.parameters)))  # type: ignore
    elif url == "http://127.0.0.1:5000/process_file":
        return MockResponse(create_response_model(process_file(data.inputs["file_inputs"].root.files, data.parameters)))  # type: ignore


def process_text(inputs: List[TextInput], parameters):
    results = [TextResponse(title=inp.text, value="processed_text.txt") for inp in inputs]
    results = BatchTextResponse(texts=results)
    return results


def process_file(inputs: List[FileInput], parameters):
    results = [
        FileResponse(title=inp.path, path="processed_image.img", file_type=FileType.IMG) for inp in inputs
    ]
    results = BatchFileResponse(files=results)
    return results


@pytest.fixture
def app():
    server = MLServer(__name__)

    class TextInputs(TypedDict):
        text_inputs: BatchTextInput

    class Parameters(TypedDict):
        pass

    class FileInputs(TypedDict):
        file_inputs: BatchFileInput

    @server.route("/process_text")
    def server_process_text(inputs: TextInputs, parameters: Parameters) -> ResponseBody:
        return ResponseBody(root=process_text(inputs["text_inputs"].texts, parameters))

    @server.route("/process_file")
    def server_process_image(inputs: FileInputs, parameters: Parameters) -> ResponseBody:
        return ResponseBody(root=process_file(inputs["file_inputs"].files, parameters))

    return server.app.test_client()


@pytest.fixture
def client():
    return MLClient("http://127.0.0.1:5000/predict")


def test_list_routes(app):
    response = app.get("/api/routes")
    assert response.status_code == 200
    assert response.json == [
        {
            "payload_schema": "/process_text/payload_schema",
            "run_task": "/process_text",
            "sample_payload": "/process_text/sample_payload",
        },
        {
            "payload_schema": "/process_file/payload_schema",
            "run_task": "/process_file",
            "sample_payload": "/process_file/sample_payload",
        },
    ]


def test_empty_list_routes():
    server = MLServer(__name__)
    app = server.app.test_client()
    response = app.get("/api/routes")
    assert response.status_code == 200
    assert response.json == []


def test_set_url(client):
    new_url = "http://localhost:8000/sentimentanalysis"
    client.set_url(new_url)
    assert client.url == new_url


def test_valid_text_request(app):
    data = {
        "inputs": {"text_inputs": {"texts": [{"text": "Sample text"}]}},
        "parameters": {},
    }

    response = app.post("/process_text", json=data)
    assert response.status_code == 200
    assert response.json == {
        "output_type": "batchtext",
        "texts": [
            {"output_type": "text", "value": "processed_text.txt", "title": "Sample text", "subtitle": None}
        ],
    }


@patch("requests.post")
def test_valid_text_request_client(mock_post, client: MLClient):
    data = {
        "inputs": {"text_inputs": {"texts": [{"text": "Sample text"}]}},
        "parameters": {},
    }

    mock_post.return_value = mock_post_request("http://127.0.0.1:5000/process_text", json=data)
    response = client.request(data["inputs"], data["parameters"])
    assert response == {
        "output_type": "batchtext",
        "texts": [
            {"output_type": "text", "value": "processed_text.txt", "title": "Sample text", "subtitle": None}
        ],
    }


def test_invalid_text_request(app):
    data = {
        "inputs": {"KEY_INVALID": {"texts": [{"text": "Sample text"}]}},
        "parameters": {},
    }
    response = app.post("/process_text", json=data)
    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "Keys mismatch. The input schema has" in response.json["error"]


def test_valid_file_request(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {},
    }

    response = app.post("/process_file", json=data)
    assert response.status_code == 200
    assert response.json == {
        "output_type": "batchfile",
        "files": [
            {
                "output_type": "file",
                "file_type": "img",
                "path": "processed_image.img",
                "title": "/path/to/image.jpg",
                "subtitle": None,
            }
        ],
    }


@patch("requests.post")
def test_valid_fike_request_client(mock_post, client):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {},
    }

    mock_post.return_value = mock_post_request("http://127.0.0.1:5000/process_file", json=data)
    response = client.request(data["inputs"], data["parameters"])

    assert response == {
        "output_type": "batchfile",
        "files": [
            {
                "output_type": "file",
                "file_type": "img",
                "path": "processed_image.img",
                "title": "/path/to/image.jpg",
                "subtitle": None,
            }
        ],
    }
def test_invalid_file_request(app):
    data = {
        "inputs": {"file_inputs": {"INVALID_KEY": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {},
    }
    response = app.post("/process_file", json=data)

    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert (
        "Field required"
        == response.json["error"][0]["msg"]
    )
