import unittest

from pydantic import ValidationError

from flask_ml.flask_ml_server.models import *


class TestFileResultModel(unittest.TestCase):
    def test_valid_file_result(self):
        data = {"file_type": "text", "path": "output_file1.txt"}
        result = FileResponse.model_validate(data)
        self.assertEqual(result.file_type, FileType.TEXT)
        self.assertEqual(result.path, "output_file1.txt")

    def test_invalid_file_result_missing_id(self):
        data = {"path": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            FileResponse(**data)  # type: ignore


class TestTextResultModel(unittest.TestCase):
    def test_valid_text_result(self):
        data = {"title": "This is the first text", "value": "output_file1.txt"}
        result = TextResponse.model_validate(data)
        self.assertEqual(result.title, "This is the first text")
        self.assertEqual(result.value, "output_file1.txt")

    def test_invalid_text_result_missing_id(self):
        data = {"result": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            TextResponse(**data)  # type: ignore


class TestResponseModel(unittest.TestCase):
    def test_valid_response_with_file_results(self):
        data = {
            "output_type": "batchfile",
            "files": [
                {"title": "This is the first text", "path": "output_file1.txt", "file_type": "text"},
                {"title": "This is the second text", "path": "output_file2.txt", "file_type": "text"},
            ],
        }
        response = ResponseBody.model_validate(data).root
        self.assertIsInstance(response, BatchFileResponse)

    def test_valid_response_with_text_results(self):
        data = {
            "output_type": "batchtext",
            "texts": [
                {"title": "This is the first text", "value": "This is the first result text value"},
                {"title": "This is the second text", "value": "This is the second result text value"},
            ],
        }
        response = ResponseBody.model_validate(data).root
        self.assertIsInstance(response, BatchTextResponse)

    def test_invalid_response_with_invalid_result_structure(self):
        data = {
            "output_type": "batchtext",
        }
        with self.assertRaises(ValidationError):
            ResponseBody.model_validate(data).root


if __name__ == "__main__":
    unittest.main()
