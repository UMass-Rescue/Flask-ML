import unittest

from pydantic import ValidationError

from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import (
    CustomInput,
    FileInput,
    RequestModel,
    TextInput,
)


class TestFileInputModel(unittest.TestCase):
    def test_valid_file_input(self):
        data = {"file_path": "/path/to/file.txt"}
        file_input = FileInput(**data)
        self.assertEqual(file_input.file_path, "/path/to/file.txt")

    def test_invalid_file_input_missing_file_path(self):
        data = {}
        with self.assertRaises(ValidationError):
            FileInput(**data)

    def test_invalid_file_input_empty_file_path(self):
        data = {"file_path": ""}
        with self.assertRaises(ValidationError):
            FileInput(**data)


class TestTextInputModel(unittest.TestCase):
    def test_valid_text_input(self):
        data = {"text": "This is a sample text"}
        text_input = TextInput(**data)
        self.assertEqual(text_input.text, "This is a sample text")

    def test_invalid_text_input_missing_text(self):
        data = {}
        with self.assertRaises(ValidationError):
            TextInput(**data)

    def test_invalid_text_input_empty_text(self):
        data = {"text": ""}
        with self.assertRaises(ValidationError):
            TextInput(**data)


class TestRequestModel(unittest.TestCase):
    def test_valid_request_with_text_inputs(self):
        data = {
            "inputs": [{"text": "First text"}, {"text": "Second text"}],
            "data_type": DataTypes.TEXT,
            "parameters": {},
        }
        request = RequestModel(**data)
        self.assertEqual(request.data_type, DataTypes.TEXT)
        self.assertEqual(len(request.inputs), 2)
        self.assertIsInstance(request.inputs[0], TextInput)
        self.assertEqual(request.inputs[0].text, "First text")

    def test_valid_request_with_file_inputs(self):
        data = {
            "inputs": [
                {"file_path": "/path/to/file1.txt"},
                {"file_path": "/path/to/file2.txt"},
            ],
            "data_type": DataTypes.IMAGE,
            "parameters": {},
        }
        request = RequestModel(**data)
        self.assertEqual(request.data_type, DataTypes.IMAGE)
        self.assertEqual(len(request.inputs), 2)
        self.assertIsInstance(request.inputs[0], FileInput)
        self.assertEqual(request.inputs[0].file_path, "/path/to/file1.txt")

    def test_valid_request_with_custom_inputs(self):
        data = {
            "inputs": [
                {"input": ["inp1", {"a": "b"}, 123]},
                {"input": {"key": "value"}},
            ],
            "data_type": DataTypes.CUSTOM,
            "parameters": {},
        }
        request = RequestModel(**data)
        self.assertEqual(request.data_type, DataTypes.CUSTOM)
        self.assertEqual(len(request.inputs), 2)
        self.assertIsInstance(request.inputs[0], CustomInput)
        self.assertEqual(request.inputs[0].input, ["inp1", {"a": "b"}, 123])
        self.assertEqual(request.inputs[1].input, {"key": "value"})

    def test_parameters_are_optional(self):
        data = {
            "inputs": [
                {"file_path": "/path/to/file1.txt"},
                {"file_path": "/path/to/file2.txt"},
            ],
            "data_type": DataTypes.IMAGE,
        }
        request = RequestModel(**data)
        self.assertEqual(request.data_type, DataTypes.IMAGE)
        self.assertEqual(len(request.inputs), 2)
        self.assertIsInstance(request.inputs[0], FileInput)
        self.assertEqual(request.inputs[0].file_path, "/path/to/file1.txt")

    def test_invalid_request_with_mismatched_data_type_and_text_input(self):
        data = {
            "inputs": [{"text": "Some text input"}],
            "data_type": DataTypes.IMAGE,
            "parameters": {},
        }
        with self.assertRaises(ValidationError):
            RequestModel(**data)

    def test_invalid_request_with_mismatched_data_type_and_file_input(self):
        data = {
            "inputs": [{"file_path": "/path/to/file.txt"}],
            "data_type": DataTypes.TEXT,
            "parameters": {},
        }
        with self.assertRaises(ValidationError):
            RequestModel(**data)

    def test_invalid_request_with_invalid_data_type(self):
        data = {
            "inputs": [{"file_path": "/path/to/file.txt"}],
            "data_type": "INVALID_TYPE",
            "parameters": {},
        }
        with self.assertRaises(ValidationError):
            RequestModel(**data)

    def test_valid_request_with_complex_parameters(self):
        data = {
            "inputs": [
                {"file_path": "/path/to/file1.txt"},
                {"file_path": "/path/to/file2.txt"},
            ],
            "data_type": DataTypes.IMAGE,
            "parameters": {"threshold": 0.8, "option": [1, 2, 3]},
        }
        request = RequestModel(**data)
        self.assertEqual(request.parameters, {"threshold": 0.8, "option": [1, 2, 3]})

    # New test case to handle invalid input in a mixed input list
    def test_invalid_mixed_inputs(self):
        data = {
            "inputs": [
                {"file_path": "/path/to/audio"},  # valid
                {"fp": "/path/to/audio2"},  # invalid, missing 'file_path'
            ],
            "data_type": DataTypes.AUDIO,
            "parameters": {},
        }
        with self.assertRaises(ValidationError) as exc_info:
            RequestModel(**data)
        self.assertIn(
            "All inputs must contain 'file_path' when data_type is AUDIO",
            str(exc_info.exception),
        )


if __name__ == "__main__":
    unittest.main()
