import requests

UNKNOWN_ERROR = "Unknown error. Please refer to the status_code field."


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

    def _validate_request_parameters(self, inputs: list[dict], data_type: str) -> None:
        """
        Validates the request parameters.
        inputs : list - the list of dictionaries containing the data to be sent to the server
        data_type : str - the type of the input data
        parameters : dict - the parameters to be sent to the server
        """
        if inputs is None:
            raise ValueError('The parameter "inputs" cannot be None')
        if data_type is None:
            raise ValueError('The parameter "data_type" cannot be None')
        if type(inputs) != list:
            raise ValueError('The parameter "inputs" is expected to be a list')
        if type(data_type) != str:
            raise ValueError('The parameter "data_type" is expected to be a string')

    def request(
        self, inputs: list[dict], data_type: str, parameters: dict = {}
    ) -> dict:
        """
        Sends a request to the server.
        inputs : list - the list of dictionaries containing the data to be sent to the server
        data_type : str - the type of the input data
        parameters : dict - the parameters to be sent to the server
        """
        self._validate_request_parameters(inputs, data_type)
        response = requests.post(
            self.url,
            json={"inputs": inputs, "data_type": data_type, "parameters": parameters},
        )
        if "application/json" not in response.headers.get("Content-Type", ""):
            return {"status": UNKNOWN_ERROR, "status_code": response.status_code}
        data = response.json()
        data["status_code"] = response.status_code
        return data
