import unittest

from pydantic import ValidationError

from flask_ml.flask_ml_server.models import (AudioResult, FileResult,
                                             ImageResult, ResponseModel,
                                             TextResult, VideoResult)


class TestFileResultModel(unittest.TestCase):
    def test_valid_file_result(self):
        data = {"file_path": "file1.txt", "result": "output_file1.txt"}
        result = FileResult(**data)
        self.assertEqual(result.file_path, "file1.txt")
        self.assertEqual(result.result, "output_file1.txt")

    def test_invalid_file_result_missing_file_path(self):
        data = {"result": "output_file1.txt"}
        with self.assertRaises(ValidationError):
            FileResult(**data)

    def test_valid_file_result_with_complex_result(self):
        data = {
            "file_path": "file1.txt",
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
                {"file_path": "file1.txt", "result": "output_file1.txt"},
                {"file_path": "file2.txt", "result": "output_file2.txt"},
            ],
        }
        response = ResponseModel(**data)
        self.assertEqual(response.status, "success")
        self.assertEqual(len(response.results), 2)
        self.assertIsInstance(response.results[0], FileResult)
        self.assertEqual(response.results[0].file_path, "file1.txt")

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
                {"file_path": "file1.txt", "result": "output_file1.txt"},
                {"text": "This is the second text", "result": "output_file2.txt"},
            ],
        }
        response = ResponseModel(**data)
        self.assertEqual(len(response.results), 2)
        self.assertIsInstance(response.results[0], FileResult)
        self.assertIsInstance(response.results[1], TextResult)

    def test_valid_response_with_missing_status(self):
        data = {"results": [{"file_path": "file1.txt", "result": "output_file1.txt"}]}
        response = ResponseModel(**data)
        self.assertEqual(response.status, "SUCCESS")

    def test_invalid_response_with_invalid_result_structure(self):
        data = {
            "status": "success",
            "results": [
                {"file_path": "file1.txt"},
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

    def test_get_response_success(self):
        response_model = ResponseModel(
            status="SUCCESS",
            results=[
                TextResult(result="Processed text", text="Sample text"),
                ImageResult(result="Processed image", file_path="/path/to/image.png"),
                AudioResult(result="Processed audio", file_path="/path/to/audio.mp3"),
                VideoResult(result="Processed video", file_path="/path/to/video.mp4"),
            ],
        )
        response = response_model.get_response()
        assert response.status_code == 200
        assert response.mimetype == "application/json"
        response_data = response.get_json()
        assert response_data["status"] == "SUCCESS"
        assert len(response_data["results"]) == 4

    def test_get_response_custom_status_code(self):
        response_model = ResponseModel(
            status="SUCCESS",
            results=[TextResult(result="Processed text", text="Sample text")],
        )
        response = response_model.get_response(status_code=201)
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
