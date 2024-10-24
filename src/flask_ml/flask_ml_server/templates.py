from typing import TypedDict, Dict, List, Union
from flask_ml.flask_ml_server import models


TYPES_TO_PARAMETER_SCHEMA = {
    float: models.FloatParameterDescriptor,
    int: models.IntParameterDescriptor,
    str: models.TextParameterDescriptor,
}


class FileInputs(TypedDict):
    file_inputs: models.BatchFileInput


class TextInputs(TypedDict):
    text_inputs: models.BatchTextInput


def parameters_to_types(parameters: Dict) -> Dict:
    return {key: type(value) for key, value in parameters.items()}


def generate_parameter_schema(parameters) -> List:
    return [generate_parameter_schema_for_value(key, value) for key, value in parameters.items()]


def generate_parameter_schema_for_value(key: str, value: Union[int, float, str]) -> models.ParameterSchema:
    return models.ParameterSchema(
        key=key,
        label=key,
        value=TYPES_TO_PARAMETER_SCHEMA[type(value)](default=value),
    )


def generate_text_response(predictions: Dict[str, str]) -> models.ResponseBody:
    result_texts = [models.TextResponse(value=p, title=k) for k, p in predictions.items()]
    response = models.BatchTextResponse(texts=result_texts)
    return models.ResponseBody(root=response)


class FileML:
    def __init__(self, parameters: Dict={}):
        self.InputType = FileInputs
        self.parameters = parameters
        parameter_types = parameters_to_types(parameters)
        self.ParameterType = TypedDict('FileMLParameters', parameter_types)  # type: ignore
        self.task_schema_func = self.file_ml_task_schema
    
    def file_ml_task_schema(self) -> models.TaskSchema:
        return models.TaskSchema(
            inputs=[
                models.InputSchema(
                    key="file_inputs", label="Provide text inputs", input_type=models.InputType.BATCHFILE
                ),
            ],
            parameters=generate_parameter_schema(self.parameters),
        )

    def generate_text_response(self, predictions: Dict[str, str]) -> models.ResponseBody:
        return generate_text_response(predictions)


class TextML:
    def __init__(self, parameters: Dict={}):
        self.InputType = TextInputs
        self.parameters = parameters
        parameter_types = parameters_to_types(parameters)
        self.ParameterType = TypedDict('TextMLParameters', parameter_types)  # type: ignore
        self.task_schema_func = self.text_ml_task_schema
    
    def text_ml_task_schema(self) -> models.TaskSchema:
        return models.TaskSchema(
            inputs=[
                models.InputSchema(
                    key="text_inputs", label="Provide text inputs", input_type=models.InputType.BATCHTEXT
                ),
            ],
            parameters=generate_parameter_schema(self.parameters),
        )

    def generate_text_response(self, predictions: Dict[str, str]) -> models.ResponseBody:
        return generate_text_response(predictions)
