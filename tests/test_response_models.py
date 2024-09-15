import unittest
from typing import Any

from models import FileResult, ResponseModel, TextResult
from pydantic import ValidationError


class TestFileResultModel(unittest.TestCase):
    def test_valid_file_result(self):
        data = {"file_name": "file1.txt", "result": "output_file1.txt"}
        result = FileResult(**data)
        self.assertEqual(result.file_name, "file1.txt")
        self.assertEqual(result.result, "output_file1.txt")

    def test_invalid_file_result_missing_file_name(self):
        data = {"result": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            FileResult(**data)

    def test_valid_file_result_with_complex_result(self):
        data = {
            "file_name": "file1.txt",
            "result": {"key": "value", "nested": [1, 2, 3]},
        }
        result = FileResult(**data)
        self.assertIsInstance(result.result, dict)
        self.assertEqual(result.result["key"], "value")
        self.assertEqual(result.result["nested"], [1, 2, 3])


class TestTextResultModel(unittest.TestCase):
    def test_valid_text_result(self):
        data = {"text": "This is the first text", "result": "output_file1.txt"}
        result = TextResult(**data)
        self.assertEqual(result.text, "This is the first text")
        self.assertEqual(result.result, "output_file1.txt")

    def test_invalid_text_result_missing_text(self):
        data = {"result": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            TextResult(**data)

    def test_valid_text_result_with_complex_result(self):
        data = {
            "text": "This is the first text",
            "result": {"key": "value", "nested": [1, 2, 3]},
        }
        result = TextResult(**data)
        self.assertIsInstance(result.result, dict)
        self.assertEqual(result.result["key"], "value")
        self.assertEqual(result.result["nested"], [1, 2, 3])


class TestResponseModel(unittest.TestCase):
    def test_valid_response_with_file_results(self):
        data = {
            "status": "success",
            "results": [
                {"file_name": "file1.txt", "result": "output_file1.txt"},
                {"file_name": "file2.txt", "result": "output_file2.txt"},
            ],
        }
        response = ResponseModel(**data)
        self.assertEqual(response.status, "success")
        self.assertEqual(len(response.results), 2)
        self.assertIsInstance(response.results[0], FileResult)
        self.assertEqual(response.results[0].file_name, "file1.txt")

    def test_valid_response_with_text_results(self):
        data = {
            "status": "success",
            "results": [
                {"text": "This is the first text", "result": "output_file1.txt"},
                {"text": "This is the second text", "result": "output_file2.txt"},
            ],
        }
        response = ResponseModel(**data)
        self.assertEqual(response.status, "success")
        self.assertEqual(len(response.results), 2)
        self.assertIsInstance(response.results[0], TextResult)
        self.assertEqual(response.results[0].text, "This is the first text")

    def test_invalid_response_with_mixed_result_types(self):
        data = {
            "status": "success",
            "results": [
                {"file_name": "file1.txt", "result": "output_file1.txt"},
                {"text": "This is the second text", "result": "output_file2.txt"},
            ],
        }
        response = ResponseModel(**data)
        self.assertEqual(len(response.results), 2)
        self.assertIsInstance(response.results[0], FileResult)
        self.assertIsInstance(response.results[1], TextResult)

    def test_valid_response_with_missing_status(self):
        data = {"results": [{"file_name": "file1.txt", "result": "output_file1.txt"}]}
        response = ResponseModel(**data)
        self.assertEqual(response.status, "SUCCESS")

    def test_invalid_response_with_invalid_result_structure(self):
        data = {
            "status": "success",
            "results": [
                {"file_name": "file1.txt"},
                {"text": "This is the second text"},
            ],
        }
        with self.assertRaises(ValidationError):
            ResponseModel(**data)

    def test_invalid_response_with_missing_results(self):
        data = {
            "status": "success",
        }
        with self.assertRaises(ValidationError):
            ResponseModel(**data)


if __name__ == "__main__":
    unittest.main()
