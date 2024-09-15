import pytest
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.models import (
    ResponseModel,
    TextResult,
    ImageResult,
    VideoResult,
    AudioResult,
    RequestModel,
)
from unittest.mock import patch


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


def mock_post_request(url, json=None, **kwargs):
    data = RequestModel(**json)
    if url == "http://127.0.0.1:5000/process_text":
        return MockResponse(process_text(data.inputs, data.parameters))
    elif url == "http://127.0.0.1:5000/process_image":
        return MockResponse(process_image(data.inputs, data.parameters))
    elif url == "http://127.0.0.1:5000/process_video":
        return MockResponse(process_video(data.inputs, data.parameters))
    elif url == "http://127.0.0.1:5000/process_audio":
        return MockResponse(process_audio(data.inputs, data.parameters))


def process_text(inputs, parameters):
    results = [TextResult(text=inp.text, result="processed_text.txt") for inp in inputs]
    response_model = ResponseModel(status="success", results=results)
    return response_model.get_response()


def process_image(inputs, parameters):
    results = [
        ImageResult(file_path=inp.file_path, result="processed_image.img")
        for inp in inputs
    ]
    response_model = ResponseModel(status="success", results=results)
    return response_model.get_response()


def process_video(inputs, parameters):
    results = [
        VideoResult(file_path=inp.file_path, result="processed_video.mp4")
        for inp in inputs
    ]
    response_model = ResponseModel(status="success", results=results)
    return response_model.get_response()


def process_audio(inputs, parameters):
    results = [
        AudioResult(file_path=inp.file_path, result="processed_audio.wav")
        for inp in inputs
    ]
    response_model = ResponseModel(status="success", results=results)
    return response_model.get_response()


@pytest.fixture
def app():
    server = MLServer(__name__)

    @server.route("/process_text", "TEXT")
    def server_process_text(inputs, parameters):
        return process_text(inputs, parameters)

    @server.route("/process_image", "IMAGE")
    def server_process_image(inputs, parameters):
        return process_image(inputs, parameters)

    @server.route("/process_video", "VIDEO")
    def server_process_video(inputs, parameters):
        return process_video(inputs, parameters)

    @server.route("/process_audio", "AUDIO")
    def server_process_audio(inputs, parameters):
        return process_audio(inputs, parameters)

    return server.app.test_client()


@pytest.fixture
def client():
    return MLClient("http://127.0.0.1:5000/predict")


def test_valid_text_request(app):
    data = {"inputs": [{"text": "Sample text"}], "data_type": "TEXT", "parameters": {}}

    response = app.post("/process_text", json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "success",
        "results": [{"text": "Sample text", "result": "processed_text.txt"}],
    }


@patch("requests.post")
def test_valid_text_request_client(mock_post, client):
    data = {"inputs": [{"text": "Sample text"}], "data_type": "TEXT", "parameters": {}}

    mock_post.return_value = mock_post_request(
        "http://127.0.0.1:5000/process_text", json=data
    )
    response = client.request(data["inputs"], data["data_type"], data["parameters"])

    assert response == [{"result": "processed_text.txt", "text": "Sample text"}]


def test_invalid_text_request(app):
    data = {
        "inputs": [{"file_path": "/path/to/file"}],
        "data_type": "TEXT",
        "parameters": {},
    }
    response = app.post("/process_text", json=data)
    # {'status': 'VALIDATION_ERROR', 'errors': [{'type': 'value_error', 'input': {'inputs': [{'file_path': 'Text to be classified'}, {'file_path': 'Another text to be classified'}], 'data_type': 'TEXT', 'parameters': {}}, 'msg': "Value error, All inputs must contain 'text' when data_type is TEXT"}], 'status_code': 400}
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
        "status": "success",
        "results": [
            {"file_path": "/path/to/image.jpg", "result": "processed_image.img"}
        ],
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

    assert response == [
        {"result": "processed_image.img", "file_path": "/path/to/image.jpg"}
    ]


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
        "status": "success",
        "results": [
            {"file_path": "/path/to/video.mp4", "result": "processed_video.mp4"}
        ],
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
        "status": "success",
        "results": [
            {"file_path": "/path/to/audio.wav", "result": "processed_audio.wav"}
        ],
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
