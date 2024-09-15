import pytest
from pydantic import ValidationError
from flask_ml.flask_ml_server.models import TextInput, FileInput, RequestModel


def test_text_input():
    text_input = TextInput(text="Sample text")
    assert text_input.text == "Sample text"


def test_file_input():
    file_input = FileInput(file_path="/path/to/file")
    assert file_input.file_path == "/path/to/file"


def test_request_model_with_text_inputs():
    json_data = {
        "inputs": [
            {"text": "Text to be classified"},
            {"text": "Another text to be classified"},
        ],
        "data_type": "TEXT",
        "parameters": {},
    }
    request_model = RequestModel(**json_data)
    assert request_model.data_type == "TEXT"
    assert len(request_model.inputs) == 2
    assert request_model.inputs[0].text == "Text to be classified"
    assert request_model.inputs[1].text == "Another text to be classified"


def test_request_model_with_file_inputs():
    json_data = {
        "inputs": [{"file_path": "/path/to/file1"}, {"file_path": "/path/to/file2"}],
        "data_type": "IMAGE",
        "parameters": {},
    }
    request_model = RequestModel(**json_data)
    assert request_model.data_type == "IMAGE"
    assert len(request_model.inputs) == 2
    assert request_model.inputs[0].file_path == "/path/to/file1"
    assert request_model.inputs[1].file_path == "/path/to/file2"


def test_request_model_with_video_inputs():
    json_data = {
        "inputs": [{"file_path": "/path/to/video1"}, {"file_path": "/path/to/video2"}],
        "data_type": "VIDEO",
        "parameters": {},
    }
    request_model = RequestModel(**json_data)
    assert request_model.data_type == "VIDEO"
    assert len(request_model.inputs) == 2
    assert request_model.inputs[0].file_path == "/path/to/video1"
    assert request_model.inputs[1].file_path == "/path/to/video2"


def test_request_model_with_audio_inputs():
    json_data = {
        "inputs": [{"file_path": "/path/to/audio1"}, {"file_path": "/path/to/audio2"}],
        "data_type": "AUDIO",
        "parameters": {},
    }
    request_model = RequestModel(**json_data)
    assert request_model.data_type == "AUDIO"
    assert len(request_model.inputs) == 2
    assert request_model.inputs[0].file_path == "/path/to/audio1"
    assert request_model.inputs[1].file_path == "/path/to/audio2"


def test_request_model_invalid_data_type():
    json_data = {
        "inputs": [{"text": "Text to be classified"}],
        "data_type": "INVALID",
        "parameters": {},
    }
    with pytest.raises(ValidationError) as exc_info:
        RequestModel(**json_data)
    assert "data_type must be one of TEXT, IMAGE, VIDEO, or AUDIO" in str(
        exc_info.value
    )


def test_request_model_mismatched_inputs_and_data_type():
    json_data = {
        "inputs": [{"file_path": "/path/to/file"}],
        "data_type": "TEXT",
        "parameters": {},
    }
    with pytest.raises(ValidationError) as exc_info:
        RequestModel(**json_data)
    assert "All inputs must contain 'text' when data_type is TEXT" in str(
        exc_info.value
    )

    json_data = {
        "inputs": [{"text": "Text to be classified"}],
        "data_type": "IMAGE",
        "parameters": {},
    }
    with pytest.raises(ValidationError) as exc_info:
        RequestModel(**json_data)
    assert "All inputs must contain 'file_path' when data_type is IMAGE" in str(
        exc_info.value
    )

    json_data = {
        "inputs": [{"file_path": "/path/to/video"}, {"text": "Text to be classified"}],
        "data_type": "VIDEO",
        "parameters": {},
    }
    with pytest.raises(ValidationError) as exc_info:
        RequestModel(**json_data)
    assert "All inputs must contain 'file_path' when data_type is VIDEO" in str(
        exc_info.value
    )

    json_data = {
        "inputs": [{"file_path": "/path/to/audio"}, {"fp": "/path/to/audio2"}],
        "data_type": "AUDIO",
        "parameters": {},
    }
    with pytest.raises(ValidationError) as exc_info:
        RequestModel(**json_data)
    assert "All inputs must contain 'file_path' when data_type is AUDIO" in str(
        exc_info.value
    )
