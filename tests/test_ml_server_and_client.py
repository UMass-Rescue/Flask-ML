from typing import List
from unittest.mock import patch

import pytest

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *

from .constants import *


def test_invalid_route_parameters():
    server = MLServer(__name__)

    with pytest.raises(ValueError, match='The parameter "rule" cannot be None'):
        server.route(None, "TEXT")

    with pytest.raises(ValueError, match='The parameter "input_type" cannot be None'):
        server.route("/test", None)

    with pytest.raises(
        ValueError, match='The parameter "rule" is expected to be a string'
    ):
        server.route(123, "TEXT")


class MockResponse:
    def __init__(self, response):
        self.status_code = response.status_code
        self.headers = {"Content-Type": "application/json"}
        self.response = response

    def json(self):
        return self.response.get_json()


def create_response_model(results):
    return ResponseModel(status="SUCCESS", results=results).get_response()


def mock_post_request(url, json=None, **kwargs):
    data = RequestModel(**json)
    if url == "http://127.0.0.1:5000/process_text":
        return MockResponse(
            create_response_model(process_text(data.inputs, data.parameters))
        )
    elif url == "http://127.0.0.1:5000/process_image":
        return MockResponse(
            create_response_model(process_image(data.inputs, data.parameters))
        )
    elif url == "http://127.0.0.1:5000/process_video":
        return MockResponse(
            create_response_model(process_video(data.inputs, data.parameters))
        )
    elif url == "http://127.0.0.1:5000/process_audio":
        return MockResponse(
            create_response_model(process_audio(data.inputs, data.parameters))
        )
    elif url == "http://127.0.0.1:5000/process_custom_input":
        return MockResponse(
            create_response_model(process_custom_input(data.inputs, data.parameters))
        )


def process_text(inputs, parameters):
    results = [TextResult(id=inp.text, result="processed_text.txt") for inp in inputs]
    results = BatchTextResult(results=results)
    return results


def process_image(inputs, parameters):
    results = [
        ImageResult(id=inp.file_path, result="processed_image.img") for inp in inputs
    ]
    results = BatchImageResult(results=results)
    return results


def process_video(inputs, parameters):
    results = [
        VideoResult(id=inp.file_path, result="processed_video.mp4") for inp in inputs
    ]
    results = BatchVideoResult(results=results)
    return results


def process_audio(inputs, parameters):
    results = [
        AudioResult(id=inp.file_path, result="processed_audio.wav") for inp in inputs
    ]
    results = BatchAudioResult(results=results)
    return results


def process_custom_input(inputs, parameters):
    # inputs is a list of models.CustomInput.
    # CustomInput has only one field called 'input'. It can be of any type.
    # Let's say our model expects "input" to contain two keys "text" and "file_path".
    results = [
        TextResult(
            id=custom_input_data.input["text"],
            result=custom_input_data.input["text"]
            + custom_input_data.input["file_path"],
        )
        for custom_input_data in inputs
    ]
    results = BatchTextResult(results=results)
    return results


@pytest.fixture
def app():
    server = MLServer(__name__)

    @server.route("/process_text", "TEXT")
    def server_process_text(inputs: List[TextInput], parameters) -> BatchTextResult:
        return process_text(inputs, parameters)

    @server.route("/process_image", "IMAGE")
    def server_process_image(inputs: List[FileInput], parameters) -> BatchImageResult:
        return process_image(inputs, parameters)

    @server.route("/process_video", "VIDEO")
    def server_process_video(inputs, parameters):
        return process_video(inputs, parameters)

    @server.route("/process_audio", "AUDIO")
    def server_process_audio(inputs, parameters):
        return process_audio(inputs, parameters)

    @server.route("/process_custom_input", "CUSTOM")
    def server_process_custom_input(inputs, parameters):
        return process_custom_input(inputs, parameters)

    return server.app.test_client()


@pytest.fixture
def client():
    return MLClient("http://127.0.0.1:5000/predict")


def test_list_routes(app):
    response = app.get("/api/routes")
    assert response.status_code == 200
    assert response.json == [
        {"rule": "/api/routes", "methods": ["GET"], "schema": None},
        {
            "rule": "/process_text",
            "methods": ["POST"],
            "schema": {
                "inputs": TEXT_INPUT_SCHEMA,
                "output": BATCH_TEXT_RESPONSE_SCHEMA,
            },
        },
        {
            "rule": "/process_image",
            "methods": ["POST"],
            "schema": {
                "inputs": FILE_INPUT_SCHEMA,
                "output": BATCH_IMAGE_RESPONSE_SCHEMA,
            },
        },
        {
            "rule": "/process_video",
            "methods": ["POST"],
            "schema": {"inputs": None, "output": None},
        },
        {
            "rule": "/process_audio",
            "methods": ["POST"],
            "schema": {"inputs": None, "output": None},
        },
        {
            "rule": "/process_custom_input",
            "methods": ["POST"],
            "schema": {"inputs": None, "output": None},
        },
    ]


def test_empty_list_routes():
    server = MLServer(__name__)
    app = server.app.test_client()
    response = app.get("/api/routes")
    assert response.status_code == 200
    assert response.json == [
        {"rule": "/api/routes", "methods": ["GET"], "schema": None}
    ]


def test_set_url(client):
    new_url = "http://localhost:8000/sentimentanalysis"
    client.set_url(new_url)
    assert client.url == new_url


def test_valid_text_request(app):
    data = {"inputs": [{"text": "Sample text"}], "data_type": "TEXT", "parameters": {}}

    response = app.post("/process_text", json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "SUCCESS",
        "results": {"results": [{"id": "Sample text", "result": "processed_text.txt"}]},
    }


@patch("requests.post")
def test_valid_text_request_client(mock_post, client):
    data = {"inputs": [{"text": "Sample text"}], "data_type": "TEXT", "parameters": {}}

    mock_post.return_value = mock_post_request(
        "http://127.0.0.1:5000/process_text", json=data
    )
    response = client.request(data["inputs"], data["data_type"], data["parameters"])
    assert response == {
        "results": [{"id": "Sample text", "result": "processed_text.txt"}]
    }


def test_invalid_text_request(app):
    data = {
        "inputs": [{"file_path": "/path/to/file"}],
        "data_type": "TEXT",
        "parameters": {},
    }
    response = app.post("/process_text", json=data)
    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "value_error" == response.json["errors"][0]["type"]
    assert (
        "Value error, All inputs must contain 'text' when data_type is TEXT"
        == response.json["errors"][0]["msg"]
    )


def test_valid_image_request(app):
    data = {
        "inputs": [{"file_path": "/path/to/image.jpg"}],
        "data_type": "IMAGE",
        "parameters": {},
    }

    response = app.post("/process_image", json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "SUCCESS",
        "results": {
            "results": [{"id": "/path/to/image.jpg", "result": "processed_image.img"}]
        },
    }


@patch("requests.post")
def test_valid_image_request_client(mock_post, client):
    data = {
        "inputs": [{"file_path": "/path/to/image.jpg"}],
        "data_type": "IMAGE",
        "parameters": {},
    }

    mock_post.return_value = mock_post_request(
        "http://127.0.0.1:5000/process_image", json=data
    )
    response = client.request(data["inputs"], data["data_type"], data["parameters"])

    assert response == {
        "results": [{"id": "/path/to/image.jpg", "result": "processed_image.img"}]
    }


def test_invalid_image_request(app):
    data = {
        "inputs": [{"text": "/path/to/file"}],
        "data_type": "IMAGE",
        "parameters": {},
    }
    response = app.post("/process_image", json=data)

    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "value_error" == response.json["errors"][0]["type"]
    assert (
        "Value error, All inputs must contain 'file_path' when data_type is IMAGE"
        == response.json["errors"][0]["msg"]
    )


def test_valid_video_request(app):
    data = {
        "inputs": [{"file_path": "/path/to/video.mp4"}],
        "data_type": "VIDEO",
        "parameters": {},
    }

    response = app.post("/process_video", json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "SUCCESS",
        "results": {
            "results": [{"id": "/path/to/video.mp4", "result": "processed_video.mp4"}]
        },
    }


def test_invalid_video_request(app):
    data = {
        "inputs": [{"text": "/path/to/file"}],
        "data_type": "VIDEO",
        "parameters": {},
    }
    response = app.post("/process_video", json=data)

    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "value_error" == response.json["errors"][0]["type"]
    assert (
        "Value error, All inputs must contain 'file_path' when data_type is VIDEO"
        == response.json["errors"][0]["msg"]
    )


def test_valid_audio_request(app):
    data = {
        "inputs": [{"file_path": "/path/to/audio.wav"}],
        "data_type": "AUDIO",
        "parameters": {},
    }

    response = app.post("/process_audio", json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "SUCCESS",
        "results": {
            "results": [{"id": "/path/to/audio.wav", "result": "processed_audio.wav"}]
        },
    }


def test_invalid_audio_request(app):
    data = {
        "inputs": [{"text": "/path/to/file"}],
        "data_type": "AUDIO",
        "parameters": {},
    }
    response = app.post("/process_audio", json=data)

    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "value_error" == response.json["errors"][0]["type"]
    assert (
        "Value error, All inputs must contain 'file_path' when data_type is AUDIO"
        == response.json["errors"][0]["msg"]
    )


def test_valid_custom_input_request(app):
    data = {
        "inputs": [
            {"input": {"text": "Sample text", "file_path": "/path/to/file.txt"}},
            {
                "input": {
                    "text": "Another text",
                    "file_path": "/path/to/another_file.txt",
                }
            },
        ],
        "data_type": "CUSTOM",
        "parameters": {},
    }

    response = app.post("/process_custom_input", json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "SUCCESS",
        "results": {
            "results": [
                {"id": "Sample text", "result": "Sample text/path/to/file.txt"},
                {
                    "id": "Another text",
                    "result": "Another text/path/to/another_file.txt",
                },
            ]
        },
    }


def test_invalid_custom_input_request(app):
    data = {
        "inputs": [{"text": "/path/to/file"}],
        "data_type": "CUSTOM",
        "parameters": {},
    }
    response = app.post("/process_custom_input", json=data)

    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "value_error" == response.json["errors"][0]["type"]
    assert (
        "Value error, All inputs must contain 'input' when data_type is CUSTOM"
        == response.json["errors"][0]["msg"]
    )
