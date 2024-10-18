import unittest

from pydantic import ValidationError

from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    BatchTextInput,
    DirectoryInput,
    FileInput,
    TextInput,
)


class TestFileInputModel(unittest.TestCase):
    def test_valid_file_input(self):
        data = {"path": "/path/to/file.txt"}
        file_input = FileInput(**data)
        self.assertEqual(file_input.path, "/path/to/file.txt")

    def test_invalid_file_input_missing_file_path(self):
        data = {}
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


class TestDirectoryInputModel(unittest.TestCase):
    def test_valid_directory_input(self):
        data = {"path": "/path/to/directory"}
        directory_input = DirectoryInput(**data)
        self.assertEqual(directory_input.path, "/path/to/directory")

    def test_invalid_directory_input_missing_directory_path(self):
        data = {}
        with self.assertRaises(ValidationError):
            DirectoryInput(**data)


class TestBatchFileInputModel(unittest.TestCase):
    def test_valid_batch_file_input(self):
        data = {"files": [{"path": "/path/to/file1.txt"}, {"path": "/path/to/file2.txt"}]}
        batch_file_input = BatchFileInput.model_validate(data)
        self.assertEqual(
            batch_file_input.model_dump()["files"],
            [{"path": "/path/to/file1.txt"}, {"path": "/path/to/file2.txt"}],
        )

    def test_invalid_batch_file_input_missing_file_paths(self):
        data = {}
        with self.assertRaises(ValidationError):
            BatchFileInput(**data)


class TestBatchTextInputModel(unittest.TestCase):
    def test_valid_batch_text_input(self):
        data = {"texts": [{"text": "This is text1"}, {"text": "This is text2"}]}
        batch_text_input = BatchTextInput.model_validate(data)
        self.assertEqual(
            batch_text_input.model_dump()["texts"], [{"text": "This is text1"}, {"text": "This is text2"}]
        )

    def test_invalid_batch_text_input_missing_texts(self):
        data = {}
        with self.assertRaises(ValidationError):
            BatchTextInput(**data)


if __name__ == "__main__":
    unittest.main()
