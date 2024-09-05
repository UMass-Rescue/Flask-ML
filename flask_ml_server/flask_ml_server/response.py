from flask import Response
import json

class ServerResponse:
    """
    The ServerResponse class is a wrapper class for server responses.
    """
    def __init__(self, message: str, status: int):
        """
        Instantiates the ServerResponse object.
        """
        self.status_message = message
        self.status_code = status

    def get_response_dict(self):
        """
        Returns the response dictionary.
        """
        return {'status': self.status_message}
    
    def get_response_json(self):
        """
        Returns the response json.
        """
        return json.dumps(self.get_response_dict())
    
    def get_response(self):
        """
        Returns the response.
        """
        return Response(response=self.get_response_json(), status=self.status_code, mimetype="application/json")


class ErrorResponse(ServerResponse):
    """
    The ErrorResponse class is a wrapper class for server error responses.
    """
    def __init__(self, message: str, status: int):
        """
        Instantiates the ErrorResponse object.
        """
        super().__init__(message, status)


class FileToTextResponse(ServerResponse):
    """
    The FileToTextResponse class is a wrapper class for responses that map a file to a text response associated with it.
    """
    def __init__(self, results: list[dict], message: str="SUCCESS", status: int=200, text_key: str="text"):
        """
        Instantiates the FileToTextResponse object.
        results : list[dict] - the list of dictionaries containing the file name and the text associated with it
        Example:
        results = [
            {
                "file_name": "file1.txt",
                text_key: "Result for file1.txt"
            },
            {
                "file_name": "file2.txt",
                text_key: "Result for file2.txt"
            }
        ]
        """
        super().__init__(message, status)
        self.results = results

    def get_response_dict(self):
        """
        Returns the response dictionary.
        """
        response_di = super().get_response_dict()
        response_di['results'] = self.results
        return response_di