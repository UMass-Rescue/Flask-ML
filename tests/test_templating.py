import json
from typing import List, TypedDict
from unittest.mock import patch

import pytest
from flask.wrappers import Response

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *
from flask_ml.flask_ml_server.templates.utils import response_body

from .constants import *

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

@pytest.fixture
def app():
    server = MLServer(__name__)

    from flask_ml.flask_ml_server.templates.no_parameters import NoParameters

    from flask_ml.flask_ml_server.templates.batchtext import create_task_schema_func, TextInputs
    @server.route("/process_texts", task_schema_func=create_task_schema_func())
    def server_process_text(inputs: TextInputs, parameters: NoParameters) -> ResponseBody:
        return ResponseBody(root=process_texts(inputs["text_inputs"].texts, parameters))
    
    from flask_ml.flask_ml_server.templates.batchfile import create_task_schema_func, FileInputs
    @server.route("/process_files", task_schema_func=create_task_schema_func())
    def server_process_files(inputs: FileInputs, parameters: NoParameters) -> ResponseBody:
        return ResponseBody(root=process_files(inputs["file_inputs"].files, parameters))

    return server.app.test_client()

def test_valid_texts_request(app):
    data = {
        "inputs": {"text_inputs": {"texts": [{"text": "Sample text"}]}},
        "parameters": {},
    }

    response = app.post("/process_texts", json=data)
    assert response.status_code == 200
    assert response.json == {
        "output_type": "batchtext",
        "texts": [
            {"output_type": "text", "value": "processed_text.txt", "title": "Sample text", "subtitle": None}
        ],
    }


def test_valid_files_request(app):
    data = {
        "inputs": {"file_inputs": {"files": [{"path": "/path/to/image.jpg"}]}},
        "parameters": {},
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

def test_response_body_func():
    result = response_body(TextResponse(value="processed_text.txt", title="Sample text", subtitle=None))
    assert result.root.output_type.name == "TEXT" # type: ignore
    assert isinstance(result.root, TextResponse)
    assert result.root.value == "processed_text.txt"
    assert result.root.title == "Sample text"
    assert result.root.subtitle is None
