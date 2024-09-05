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


class TextResponse(ServerResponse):
    """
    The TextResponse class helps generate a response object for textual responses.
    """
    def __init__(self, results: list[dict], message: str="SUCCESS", status: int=200):
        """
        Instantiates the TextResponse object.
        results : list - the list of dictionaries containing the file name or text and the result text associated with it
        Example:
        results = [
            {
                "file_name": "file1.txt",
                "result": "Result for file1.txt"
            },
            {
                "file_name": "file2.txt",
                "result": "Result for file2.txt"
            }
        ]
        or 
        results = [
            {
                "text": "This is the first text",
                "result": "Result for first text"
            },
            {
                "text": "This is the second text",
                "result": "Result for second text"
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