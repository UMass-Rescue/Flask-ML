from typing import Any, Dict, List, Sequence, Union

from flask import Response
from pydantic import BaseModel, Field, field_validator, model_validator


class MLInput(BaseModel):
    pass


class TextInput(MLInput):
    text: str = Field(..., description="Text to be processed", min_length=1)


class FileInput(MLInput):
    file_path: str = Field(
        ..., description="Path of the file to be processed", min_length=1
    )


class CustomInput(MLInput):
    input: Any = Field(..., description="Custom input")


class RequestModel(BaseModel):
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

    inputs: List[Union[TextInput, FileInput, CustomInput]] = Field(
        ...,
        description="List of input items to be processed",
        min_length=1,
    )
    data_type: str = Field(
        ...,
        description="Type of the data, should be TEXT, IMAGE, VIDEO, AUDIO, or CUSTOM",
        min_length=1,
    )
    parameters: Dict = Field(
        default_factory=dict, description="Additional parameters for the ML model"
    )

    @field_validator("data_type", mode="before")
    def check_data_type(cls, v):
        if v not in {"TEXT", "IMAGE", "VIDEO", "AUDIO", "CUSTOM"}:
            raise ValueError(
                "data_type must be one of TEXT, IMAGE, VIDEO, AUDIO, or CUSTOM"
            )
        return v

    @model_validator(mode="before")
    def check_inputs_match_data_type(cls, values):
        data_type = values.get("data_type")
        inputs = values.get("inputs", [])

        if data_type == "TEXT":
            if not all(isinstance(item.get("text"), str) for item in inputs):
                raise ValueError(
                    "All inputs must contain 'text' when data_type is TEXT"
                )
        elif data_type in ["IMAGE", "VIDEO", "AUDIO"]:
            if not all(isinstance(item.get("file_path"), str) for item in inputs):
                raise ValueError(
                    f"All inputs must contain 'file_path' when data_type is {data_type}"
                )
        elif data_type == "CUSTOM":
            if not all("input" in item for item in inputs):
                raise ValueError(
                    "All inputs must contain 'input' when data_type is CUSTOM"
                )
        return values


class MLResult(BaseModel):
    id: str = Field(..., description="The ID of the result", min_length=1)


class FileResult(MLResult):
    result: str = Field(..., description="Path of the result file.", min_length=1)


class ImageResult(FileResult):
    pass


class VideoResult(FileResult):
    pass


class AudioResult(FileResult):
    pass


class TextResult(MLResult):
    result: str = Field(..., description="The result text.", min_length=1)


class BatchTextResult(BaseModel):
    results: List[TextResult] = Field(
        ..., description="List of text results", min_length=1
    )


class BatchImageResult(BaseModel):
    results: List[ImageResult] = Field(
        ..., description="List of image results", min_length=1
    )


class BatchAudioResult(BaseModel):
    results: List[AudioResult] = Field(
        ..., description="List of audio results", min_length=1
    )


class BatchVideoResult(BaseModel):
    results: List[VideoResult] = Field(
        ..., description="List of video results", min_length=1
    )


class BaseResponseModel(BaseModel):
    """
    Base model for response models.
    Methods:
        model_dump_json() -> str:
            Returns the JSON representation of the model.
    """

    def get_response(self, status_code: int = 200):
        return Response(
            response=self.model_dump_json(),
            status=status_code,
            mimetype="application/json",
        )


class ResponseModel(BaseResponseModel):
    """
    Model representing the results from an ML model.
    Attributes:
        status (str): The status of the operation, e.g., 'success'
        results (List[MLResult]): List of results.
    Methods:
        get_response(status_code: int = 200) -> Response:
            Returns a Flask Response object with the JSON representation of the model.
    """

    status: str = Field(
        default="SUCCESS",
        description="The status of the operation, e.g., 'SUCCESS'",
        min_length=1,
    )
    results: Union[
        BatchTextResult, BatchVideoResult, BatchAudioResult, BatchImageResult
    ] = Field(
        ...,
        description="List of results",
    )


class ErrorResponseModel(BaseResponseModel):
    """
    Model representing an error response.
    Attributes:
        status (str): The status of the operation, e.g., 'error'
        detail (str): Details about the error that occurred.
    Methods:
        get_response(status_code: int = 400) -> Response:
            Returns a Flask Response object with the JSON representation of the model.
    """

    status: str = Field(
        default="ERROR",
        description="The status of the operation, e.g., 'ERROR'",
        min_length=1,
    )
    errors: List = Field(..., description="Details about the error that occurred")

    def get_response(self, status_code: int = 400):
        return super().get_response(status_code)
