from typing import Any, Dict, Union

import requests

from flask_ml.flask_ml_server.models import Input, RequestBody, ResponseBody

UNKNOWN_ERROR = "Unknown error. Please refer to the status field."


class MLClient:
    """
    The MLClient class is a wrapper class for making requests to the MLServer object.
    """

    def __init__(self, url: str):
        """
        Instantiates the MLClient object.
        url : str - the URL of the server
        Ex: http://127.0.0.1:5000
        """
        self.url = url

    def set_url(self, url: str):
        """
        Sets the URL of the server.
        url : str - the URL of the server
        """
        self.url = url

    def request(self, inputs: Union[Dict[str, Input], Dict[str, Dict]], parameters: Dict[str, Any] = {}):
        """
        Sends a request to the server.
        inputs : list - the list of dictionaries containing the data to be sent to the server
        data_type : str - the type of the input data
        parameters : dict - the parameters to be sent to the server
        """
        request_model = RequestBody.model_validate({"inputs": inputs, "parameters": parameters})
        response = requests.post(
            self.url,
            json=request_model.model_dump(),
        )
        if "application/json" not in response.headers.get("Content-Type", ""):
            return {
                "status": f"Unknown error. status_code={str(response.status_code)}",
                "errors": [{"msg": UNKNOWN_ERROR}],
            }
        if response.status_code != 200:
            return response.json()
        response_model = ResponseBody(**response.json())
        return response_model.model_dump(mode="json")
