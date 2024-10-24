import json
from typing import List, TypedDict
from unittest.mock import patch

import pytest
from flask.wrappers import Response

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *
from tests.conftest import MockResponse, mock_post_request

from .constants import *

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
            "payload_schema": "/process_texts/payload_schema",
            "run_task": "/process_texts",
            "sample_payload": "/process_texts/sample_payload",
        },
        {
            "payload_schema": "/process_file/payload_schema",
            "run_task": "/process_file",
            "sample_payload": "/process_file/sample_payload",
        },
        {
            "payload_schema": "/process_files/payload_schema",
            "run_task": "/process_files",
            "sample_payload": "/process_files/sample_payload",
        },
        {
            "payload_schema": "/process_invalid/payload_schema",
            "run_task": "/process_invalid",
            "sample_payload": "/process_invalid/sample_payload",
        },
        {
            "order": 0,
            "payload_schema": "/process_text_with_schema/payload_schema",
            "run_task": "/process_text_with_schema",
            "sample_payload": "/process_text_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_text_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_texts_with_schema/payload_schema",
            "run_task": "/process_texts_with_schema",
            "sample_payload": "/process_texts_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_texts_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_file_with_schema/payload_schema",
            "run_task": "/process_file_with_schema",
            "sample_payload": "/process_file_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_file_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_files_with_schema/payload_schema",
            "run_task": "/process_files_with_schema",
            "sample_payload": "/process_files_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_files_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_invalid_with_schema/payload_schema",
            "run_task": "/process_invalid_with_schema",
            "sample_payload": "/process_invalid_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_invalid_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_directory_and_enum_parameter_with_schema/payload_schema",
            "run_task": "/process_directory_and_enum_parameter_with_schema",
            "sample_payload": "/process_directory_and_enum_parameter_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_directory_and_enum_parameter_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_directories_and_ranged_int_parameter_with_schema/payload_schema",
            "run_task": "/process_directories_and_ranged_int_parameter_with_schema",
            "sample_payload": "/process_directories_and_ranged_int_parameter_with_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_directories_and_ranged_int_parameter_with_schema/task_schema",
        },
        {
            "order": 0,
            "payload_schema": "/process_text_input_with_text_area_schema/payload_schema",
            "run_task": "/process_text_input_with_text_area_schema",
            "sample_payload": "/process_text_input_with_text_area_schema/sample_payload",
            "short_title": "",
            "task_schema": "/process_text_input_with_text_area_schema/task_schema",
        }
    ]


def test_empty_list_routes():
    server = MLServer(__name__)
    app = server.app.test_client()
    response = app.get("/api/routes")
    assert response.status_code == 200
    assert response.json == []


def test_payload_schema(app):
    response = app.get("/process_file/payload_schema")
    assert response.status_code == 200
    assert "$defs" in response.json


def test_sample_payload(app):
    response = app.get("process_files/sample_payload")
    assert response.status_code == 200
    assert response.json == {
        "inputs": {
            "file_inputs": {
                "files": [{"path": "/Users/path/to/file1"}, {"path": "/Users/path/to/file2"}]
            }
        },
        "parameters": {"param1": 1.0},
    }


def test_payload_schema_with_task_schema(app):
    response = app.get("/process_files_with_schema/payload_schema")
    assert response.status_code == 200
    assert "$defs" in response.json


def test_sample_payload_with_task_schema(app):
    response = app.get(f"/process_text_with_schema/sample_payload")
    assert response.status_code == 200
    assert response.json == {
        "inputs": {"text_input": {"text": "A sample piece of text"}},
        "parameters": {"param1": "default"},
    }


def test_task_schema(app):
    response = app.get("/process_files_with_schema/task_schema")
    assert response.status_code == 200
    assert response.json == {
        "inputs": [
            {
                "input_type": "batchfile",
                "key": "file_inputs",
                "label": "Batch File Inputs",
                "subtitle": "",
            }
        ],
        "parameters": [
            {
                "key": "param1",
                "label": "Ranged Float Parameter",
                "subtitle": "",
                "value": {
                    "parameter_type": "ranged_float",
                    "default": 0.0,
                    "range": {"min": 0.0, "max": 1.0},
                },
            }
        ],
    }

def test_valid_file_request_for_endpoint_with_task_schema(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }
    response = app.post("/process_files_with_schema", json=data)
    assert response.status_code == 200
    assert response.json ==  {
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


def test_bad_request_input_validation_error_for_endpoint_with_schema(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"INCORRECT KEY": 0.0},
    }

    response = app.post("/process_files_with_schema", json=data)
    assert response.status_code == 400
    assert "Keys mismatch." in response.json["error"]


def test_bad_request_param_validation_error_for_endpoint_with_schema(app):
    data = {
        "inputs": {"INCORRECT_KEY": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }

    response = app.post("/process_files_with_schema", json=data)
    assert response.status_code == 400
    assert "Keys mismatch." in response.json["error"]


def test_invalid_request_param_validation_error_for_endpoint_with_schema(app):
    data = {
        "inputs": {"file_inputs": {"incorret_key": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }

    response = app.post("/process_files_with_schema", json=data)
    assert response.status_code == 400
    assert "Field required" in response.json["error"][0]["msg"]


def test_set_url(client):
    new_url = "http://localhost:8000/sentimentanalysis"
    client.set_url(new_url)
    assert client.url == new_url


def test_valid_text_request(app):
    data = {
        "inputs": {"text_inputs": {"texts": [{"text": "Sample text"}]}},
        "parameters": {"param1": 0},
    }

    response = app.post("/process_texts", json=data)
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
        "parameters": {"param1": 0},
    }

    mock_post.return_value = mock_post_request("http://127.0.0.1:5000/process_texts", json=data)
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
        "parameters": {"param1": "Sample value for parameter"},
    }
    response = app.post("/process_texts", json=data)
    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "Keys mismatch. The input schema has" in response.json["error"]


def test_valid_file_request(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }

    response = app.post("/process_files", json=data)
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
def test_valid_file_request_client(mock_post, client):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }

    mock_post.return_value = mock_post_request("http://127.0.0.1:5000/process_files", json=data)
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


@patch("requests.post")
def test_invalid_response_not_json(mock_post, client):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }
    mock_post.return_value = mock_post_request("http://127.0.0.1:5000/process_files", json=data)
    mock_post.return_value.headers = {"Content-Type": "text/html"}
    response = client.request(data["inputs"], data["parameters"])
    assert "Unknown error" in response["status"]
    assert "errors" in response
    assert "Unknown error" in response["errors"][0]["msg"]


@patch("requests.post")
def test_400_response(mock_post, client):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }
    mock_post.return_value = MockResponse(
        response=Response(response=json.dumps({"status": "failed"}), status=400, mimetype="application/json")
    )
    mock_post.return_value.status_code = 400
    response = client.request(data["inputs"], data["parameters"])
    assert {"status": "failed"} == response

@patch("requests.post")
def test_500_response(mock_post, client):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }
    mock_post.return_value = MockResponse(
        response=Response(response=json.dumps({"status": "internal server error"}), status=500, mimetype="application/json")
    )
    mock_post.return_value.status_code = 500
    response = client.request(data["inputs"], data["parameters"])
    assert {"status": "internal server error"} == response


def test_invalid_file_request(app):
    data = {
        "inputs": {"file_inputs": {"INVALID_KEY": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {},
    }
    response = app.post("/process_files", json=data)

    assert response.status_code == 400
    assert "VALIDATION_ERROR" == response.json["status"]
    assert "Field required" == response.json["error"][0]["msg"]

def test_500_error_handling_for_endpoint_without_schema(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }
    response = app.post("/process_invalid", json=data)
    assert response.status_code == 500
    assert response.json == {'status': "SERVER_ERROR", 'error': "Exception('Internal Server Error')"}

def test_500_error_handling_for_endpoint_with_schema(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {"param1": 0.0},
    }
    response = app.post("/process_invalid_with_schema", json=data)
    assert response.status_code == 500
    assert response.json == {'status': "SERVER_ERROR", 'error': "Exception('Internal Server Error')"}