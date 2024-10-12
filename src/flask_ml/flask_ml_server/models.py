from flask import Response
from pydantic import model_validator
from ._generated_models import *


class RequestModel(RequestModelGenerated):
    """
    Represents a request model for processing inputs with an ML model.
    Attributes:
        inputs (List[MLInput]): List of input items to be processed.
        data_type (str): Type of the data, should be TEXT, IMAGE, VIDEO, or AUDIO.
        parameters (Dict): Additional parameters for the ML model.
    Methods:
        check_data_type(v): Validates the data_type attribute.
        check_inputs_match_data_type(values): Validates that the inputs match the data_type.
    Raises:
        ValueError: If the data_type is not one of TEXT, IMAGE, VIDEO, or AUDIO.
        ValueError: If the inputs do not match the data_type.
    """

    @model_validator(mode="before")
    def check_inputs_match_data_type(cls, values):
        data_type = values.get("data_type")
        inputs = values.get("inputs", [])
        
        if data_type == DataTypes.TEXT.name:
            if not all(isinstance(item.get("text"), str) for item in inputs):
                raise ValueError("All inputs must contain 'text' when data_type is TEXT")
        elif data_type in [DataTypes.IMAGE.name, DataTypes.VIDEO.name, DataTypes.AUDIO.name]:
            if not all(isinstance(item.get("file_path"), str) for item in inputs):
                raise ValueError(f"All inputs must contain 'file_path' when data_type is {data_type}")
        elif data_type == DataTypes.CUSTOM.name:
            if not all("input" in item for item in inputs):
                raise ValueError("All inputs must contain 'input' when data_type is CUSTOM")
        else:
            raise ValueError(f"FATAL: invalid data_type: {data_type}")
        return values


def create_flask_response(response: ResponseModel, status_code: int = 200) -> Response:
    return Response(
        response=response.model_dump_json(),
        status=status_code,
        mimetype="application/json",
    )
