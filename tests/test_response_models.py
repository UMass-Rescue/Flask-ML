import unittest

from pydantic import ValidationError

from flask_ml.flask_ml_server.models import *


class TestFileResultModel(unittest.TestCase):
    def test_valid_file_result(self):
        data = {"id": "file1.txt", "result": "output_file1.txt"}
        result = FileResult(**data)
        self.assertEqual(result.id, "file1.txt")
        self.assertEqual(result.result, "output_file1.txt")

    def test_invalid_file_result_missing_id(self):
        data = {"result": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            FileResult(**data)

    def test_invalid_file_result_with_complex_result(self):
        data = {
            "id": "file1.txt",
            "result": {"key": "value", "nested": [1, 2, 3]},
        }
        with self.assertRaises(ValidationError):
            TextResult(**data)


class TestTextResultModel(unittest.TestCase):
    def test_valid_text_result(self):
        data = {"id": "This is the first text", "result": "output_file1.txt"}
        result = TextResult(**data)
        self.assertEqual(result.id, "This is the first text")
        self.assertEqual(result.result, "output_file1.txt")

    def test_invalid_text_result_missing_id(self):
        data = {"result": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            TextResult(**data)

    def test_invalid_text_result_with_complex_result(self):
        data = {
            "id": "This is the first text",
            "result": {"key": "value", "nested": [1, 2, 3]},
        }
        with self.assertRaises(ValidationError):
            TextResult(**data)


class TestResponseModel(unittest.TestCase):
    def test_valid_response_with_file_results(self):
        data = {
            "status": "success",
            "results": BatchImageResult(
                results=[
                    {"id": "file1.txt", "result": "output_file1.png"},
                    {"id": "file2.txt", "result": "output_file2.png"},
                ]
            ),
        }
        response = ResponseModel(**data)
        self.assertEqual(response.status, "success")
        self.assertEqual(len(response.results.results), 2)
        self.assertIsInstance(response.results.results[0], ImageResult)
        self.assertEqual(response.results.results[0].id, "file1.txt")

    def test_valid_response_with_text_results(self):
        data = {
            "status": "success",
            "results": {
                "results": [
                    {"id": "This is the first text", "result": "output_file1.txt"},
                    {"id": "This is the second text", "result": "output_file2.txt"},
                ]
            },
        }
        response = ResponseModel(**data)
        self.assertEqual(response.status, "success")
        self.assertEqual(len(response.results.results), 2)
        self.assertIsInstance(response.results.results[0], TextResult)
        self.assertEqual(response.results.results[0].id, "This is the first text")

    def test_valid_response_with_missing_status(self):
        data = {
            "results": {"results": [{"id": "file1.txt", "result": "output_file1.txt"}]}
        }
        response = ResponseModel(**data)
        self.assertEqual(response.status, "SUCCESS")

    def test_invalid_response_with_invalid_result_structure(self):
        data = {
            "status": "success",
            "results": {
                "results": [
                    {"id": "file1.txt"},
                    {"id": "This is the second text"},
                ]
            },
        }
        with self.assertRaises(ValidationError):
            ResponseModel(**data)

    def test_invalid_response_with_missing_results(self):
        data = {
            "status": "success",
        }
        with self.assertRaises(ValidationError):
            ResponseModel(**data)

    def test_get_response_success(self):
        response_model = ResponseModel(
            status="SUCCESS",
            results=BatchTextResult(
                results=[
                    TextResult(result="Processed text", id="Sample text"),
                    TextResult(result="Processed text2", id="Sample text2"),
                ]
            ),
        )
        response = create_flask_response(response_model)
        assert response.status_code == 200
        assert response.mimetype == "application/json"
        response_data = response.get_json()
        assert response_data["status"] == "SUCCESS"
        assert len(response_data["results"]["results"]) == 2

    def test_get_response_custom_status_code(self):
        response_model = ResponseModel(
            status="SUCCESS",
            results=BatchTextResult(
                results=[TextResult(result="Processed text", id="Sample text")]
            ),
        )
        response = create_flask_response(response_model, status_code=201)
        assert response.status_code == 201
        assert response.mimetype == "application/json"
        response_data = response.get_json()
        assert response_data["status"] == "SUCCESS"
        assert len(response_data["results"]) == 1

    def test_response_model_validation_error(self):
        with self.assertRaises(ValidationError):
            ResponseModel(status="SUCCESS", results=[{"result": "Invalid result type"}])


if __name__ == "__main__":
    unittest.main()
