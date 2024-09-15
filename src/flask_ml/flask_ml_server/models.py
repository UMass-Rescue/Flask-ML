from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Union, Dict, Any


class TextInput(BaseModel):
    text: str = Field(..., description="Text to be processed")


class FileInput(BaseModel):
    file_path: str = Field(..., description="Path of the file to be processed")


class RequestModel(BaseModel):
    inputs: List[Union[TextInput, FileInput]] = Field(
        ..., description="List of input items to be processed"
    )
    data_type: str = Field(
        ..., description="Type of the data, should be TEXT, IMAGE, VIDEO, or AUDIO"
    )
    parameters: Dict = Field(
        default_factory=dict, description="Additional parameters for the ML model"
    )

    @field_validator("data_type", mode="before")
    def check_data_type(cls, v):
        if v not in {"TEXT", "IMAGE", "VIDEO", "AUDIO"}:
            raise ValueError("data_type must be one of TEXT, IMAGE, VIDEO, or AUDIO")
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
        else:
            if not all(isinstance(item.get("file_path"), str) for item in inputs):
                raise ValueError(
                    f"All inputs must contain 'file_path' when data_type is {data_type}"
                )
        return values


class FileResult(BaseModel):
    file_path: str = Field(..., description="Path of the file associated with the result")
    result: Any = Field(..., description="The result, which can be any JSON-serializable object")


class TextResult(BaseModel):
    text: str = Field(..., description="The text content associated with the result")
    result: Any = Field(..., description="The result, which can be any JSON-serializable object")


class ResponseModel(BaseModel):
    status: str = Field(..., description="The status of the operation, e.g., 'success'")
    results: List[Union[FileResult, TextResult]] = Field(..., description="List of results, each either a file or text with its result")
