from typing import Any, Callable, Dict, List, Mapping, Union, get_type_hints

from pydantic import BaseModel
from typing_extensions import assert_never

from flask_ml.flask_ml_server import models
from flask_ml.flask_ml_server.errors import BadRequestError
from flask_ml.flask_ml_server.MLServer import TaskSchema
from flask_ml.flask_ml_server.models import (
    BatchDirectoryInput,
    BatchFileInput,
    BatchTextInput,
    DirectoryInput,
    EnumParameterDescriptor,
    FileInput,
    FloatParameterDescriptor,
    Input,
    InputType,
    IntParameterDescriptor,
    ParameterType,
    RangedFloatParameterDescriptor,
    RangedIntParameterDescriptor,
    RequestBody,
    ResponseBody,
    TextInput,
    TextParameterDescriptor,
)


def validate_data_is_dict(data_: Any, key="Request body"):
    if not isinstance(data_, dict):
        raise BadRequestError(
            f"{key} must be a valid JSON dictionary. Provided request body {data_} is not a valid dictionary. Call /api/routes to see how to use the API."
        )
    data: dict[str, Any] = data_
    return data


def validate_data_has_keys(data_: Any, keys: List[str]):
    data = validate_data_is_dict(data_)
    if set(data.keys()) != set(keys):
        raise BadRequestError(
            f"Request body must be a valid JSON dictionary and contain exactly the keys in {keys=}. Provided request body {data_} is not a valid. Call /api/routes to see how to use the API."
        )


def input_from_data(input_type: InputType, data: Dict[str, Any]):
    match input_type:
        case InputType.FILE:
            return FileInput(**data)
        case InputType.DIRECTORY:
            return DirectoryInput(**data)
        case InputType.TEXT:
            return TextInput(**data)
        case InputType.TEXTAREA:
            return TextInput(**data)
        case InputType.BATCHFILE:
            return BatchFileInput(**data)
        case InputType.BATCHTEXT:
            return BatchTextInput(**data)
        case InputType.BATCHDIRECTORY:
            return BatchDirectoryInput(**data)
        case _: # pragma: no cover
            assert_never(input_type)


def schema_get_inputs(schema: TaskSchema, data_: Dict[str, Any]):
    json_inputs = validate_data_is_dict(data_, "inputs")
    input_schema = schema.inputs
    input_keys_to_input_type = {inputt.key: inputt.input_type for inputt in input_schema}
    input_keys = set(input_keys_to_input_type.keys())
    json_keys = set(json_inputs.keys())
    if input_keys != json_keys:
        raise BadRequestError(
            f"Keys mismatch. The input schema has {input_keys=} while your json data has {json_keys=}. Ensure the request body contains all keys under the key 'inputs'. Call /api/routes to see how to use the API."
        )
    return {key: input_from_data(input_keys_to_input_type[key], json_inputs[key]) for key in input_keys}


def schema_get_parameters(schema: TaskSchema, data_: Dict[str, Any]) -> Dict[str, Union[str, int, float]]:
    json_parameters = validate_data_is_dict(data_, "parameters")
    parameter_schema = schema.parameters
    parameter_keys = set([parameter.key for parameter in parameter_schema])
    json_keys = set(json_parameters.keys())
    if parameter_keys != json_keys:
        raise BadRequestError(
            f"Keys mismatch. The parameter schema has {parameter_keys=} while your json data has {json_keys=}. Ensure the request body contains all keys under the key 'parameters'. Call /api/routes to see how to use the API."
        )
    return {key: json_parameters[key] for key in parameter_keys}


def schema_get_sample_payload(schema: TaskSchema) -> RequestBody:
    input_schema = schema.inputs
    parameter_schema = schema.parameters

    inputs: Dict[str, Input] = {}
    parameters = {}
    for input_schema in input_schema:
        input_type = input_schema.input_type
        match input_type:
            case InputType.FILE:
                inputs[input_schema.key] = Input(root=FileInput(path="/Users/path/to/file"))
            case InputType.DIRECTORY:
                inputs[input_schema.key] = Input(root=DirectoryInput(path="/Users/path/to/folder"))
            case InputType.TEXT:
                inputs[input_schema.key] = Input(root=TextInput(text="A sample piece of text"))
            case InputType.TEXTAREA:
                inputs[input_schema.key] = Input(
                    root=TextInput(
                        text="A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long."
                    )
                )
            case InputType.BATCHFILE:
                inputs[input_schema.key] = Input(
                    root=BatchFileInput(
                        files=[
                            FileInput(path="/Users/path/to/file1"),
                            FileInput(path="/Users/path/to/file2"),
                        ]
                    )
                )
            case InputType.BATCHTEXT:
                inputs[input_schema.key] = Input(
                    root=BatchTextInput(
                        texts=[
                            TextInput(text="A sample piece of text 1"),
                            TextInput(text="A sample piece of text 2"),
                        ]
                    )
                )
            case InputType.BATCHDIRECTORY:
                inputs[input_schema.key] = Input(
                    root=BatchDirectoryInput(
                        directories=[
                            DirectoryInput(path="/Users/path/to/folder1"),
                            DirectoryInput(path="/Users/path/to/folder2"),
                        ]
                    )
                )
            case _: # pragma: no cover
                assert_never(input_type)
    for parameter_schema in parameter_schema:
        match parameter_schema.value:
            case RangedFloatParameterDescriptor():
                parameters[parameter_schema.key] = parameter_schema.value.range.min
            case FloatParameterDescriptor():
                parameters[parameter_schema.key] = parameter_schema.value.default
            case EnumParameterDescriptor():
                parameters[parameter_schema.key] = parameter_schema.value.enum_vals[0].key
            case TextParameterDescriptor():
                parameters[parameter_schema.key] = parameter_schema.value.default
            case RangedIntParameterDescriptor():
                parameters[parameter_schema.key] = parameter_schema.value.range.min
            case IntParameterDescriptor():
                parameters[parameter_schema.key] = parameter_schema.value.default
            case _: # pragma: no cover
                assert_never(parameter_schema.value)
    return RequestBody(inputs=inputs, parameters=parameters)


def resolve_input_sample(input_type: Any) -> Input:
    # TODO: Add a parameterized test for this function
    match input_type:
        case models.FileInput:
            return Input(root=FileInput(path="/Users/path/to/file"))
        case models.DirectoryInput:
            return Input(root=DirectoryInput(path="/Users/path/to/folder"))
        case models.TextInput:
            return Input(root=TextInput(text="A sample piece of text"))
        case models.BatchFileInput:
            return Input(
                root=BatchFileInput(
                    files=[
                        FileInput(path="/Users/path/to/file1"),
                        FileInput(path="/Users/path/to/file2"),
                    ]
                )
            )
        case models.BatchTextInput:
            return Input(
                root=BatchTextInput(
                    texts=[
                        TextInput(text="A sample piece of text 1"),
                        TextInput(text="A sample piece of text 2"),
                    ]
                )
            )
        case models.BatchDirectoryInput:
            return Input(
                root=BatchDirectoryInput(
                    directories=[
                        DirectoryInput(path="/Users/path/to/folder1"),
                        DirectoryInput(path="/Users/path/to/folder2"),
                    ]
                )
            )
        case _: # pragma: no cover
            assert_never(input_type)


def is_typeddict(cls):
    return isinstance(cls, type) and hasattr(cls, "__annotations__")


def ensure_ml_func_parameters_are_typed_dict(ml_function: Callable[[Any, Any], ResponseBody]):
    hints = get_type_hints(ml_function)
    if not is_typeddict(hints["inputs"]):
        raise BadRequestError(f"Inputs must be a TypedDict")
    if not is_typeddict(hints["parameters"]):
        raise BadRequestError(f"Parameters must be a TypedDict")


def ensure_ml_func_hinting_and_task_schemas_are_valid(
    ml_function: Callable[[Any, Any], ResponseBody], task_schema: TaskSchema
):
    hints = get_type_hints(ml_function)
    input_type_hints: Mapping[str, BaseModel] = get_type_hints(hints["inputs"])
    parameters_type_hints: Mapping[str, Union[str, int, float]] = get_type_hints(hints["parameters"])

    input_schema = task_schema.inputs
    parameters_schema = task_schema.parameters

    input_schema_input_key_to_input_type = {inputt.key: inputt.input_type for inputt in input_schema}
    parameters_schema_key_to_parameter_type = {
        parameter.key: parameter.value.parameter_type for parameter in parameters_schema
    }

    assert list(input_type_hints.keys()) == list(
        input_schema_input_key_to_input_type.keys()
    ), f"Input schema and Typed Dict for inputs must have the same keys. Input schema keys: {input_schema_input_key_to_input_type.keys()} | Typed Dict keys: {input_type_hints.keys()}"
    assert list(parameters_type_hints.keys()) == list(
        parameters_schema_key_to_parameter_type.keys()
    ), f"Parameter schema and Typed Dict for parameters must have the same keys. Parameter schema keys: {parameters_schema_key_to_parameter_type.keys()} | Typed Dict keys: {parameters_type_hints.keys()}"

    for key in input_schema_input_key_to_input_type:
        input_type_hint = input_type_hints[key]
        input_type = input_schema_input_key_to_input_type[key]
        match input_type:
            case InputType.FILE:
                assert (
                    input_type_hint is FileInput
                ), f"For key {key}, the input type is InputType.FILE, but the TypeDict hint is {input_type_hint}. Change to FileInput."
            case InputType.DIRECTORY:
                assert (
                    input_type_hint is DirectoryInput
                ), f"For key {key}, the input type is InputType.DIRECTORY, but the TypeDict hint is {input_type_hint}. Change to DirectoryInput."
            case InputType.TEXT:
                assert (
                    input_type_hint is TextInput
                ), f"For key {key}, the input type is InputType.TEXT, but the TypeDict hint is {input_type_hint}. Change to TextInput."
            case InputType.TEXTAREA:
                assert (
                    input_type_hint is TextInput
                ), f"For key {key}, the input type is InputType.TEXTAREA, but the TypeDict hint is {input_type_hint}. Change to TextInput."
            case InputType.BATCHFILE:
                assert (
                    input_type_hint is BatchFileInput
                ), f"For key {key}, the input type is InputType.BATCHFILE, but the TypeDict hint is {input_type_hint}. Change to BatchFileInput."
            case InputType.BATCHTEXT:
                assert (
                    input_type_hint is BatchTextInput
                ), f"For key {key}, the input type is InputType.BATCHTEXT, but the TypeDict hint is {input_type_hint}. Change to BatchTextInput."
            case InputType.BATCHDIRECTORY:
                assert (
                    input_type_hint is BatchDirectoryInput
                ), f"For key {key}, the input type is InputType.BATCHDIRECTORY, but the TypeDict hint is {input_type_hint}. Change to BatchDirectoryInput."
            case _: # pragma: no cover
                assert_never(input_type)

    for key in parameters_schema_key_to_parameter_type:
        parameter_type_hint = parameters_type_hints[key]
        parameter_type: ParameterType = parameters_schema_key_to_parameter_type[key]  # type: ignore
        match parameter_type:
            case ParameterType.RANGED_FLOAT:
                assert (
                    parameter_type_hint is float
                ), f"For key {key}, the parameter type is ParameterType.RANGED_FLOAT, but the TypeDict hint is {parameter_type_hint}. Change to float."
            case ParameterType.FLOAT:
                assert (
                    parameter_type_hint is float
                ), f"For key {key}, the parameter type is ParameterType.FLOAT, but the TypeDict hint is {parameter_type_hint}. Change to float."
            case ParameterType.ENUM:
                assert (
                    parameter_type_hint is str
                ), f"For key {key}, the parameter type is ParameterType.ENUM, but the TypeDict hint is {parameter_type_hint}. Change to str."
            case ParameterType.TEXT:
                assert (
                    parameter_type_hint is str
                ), f"For key {key}, the parameter type is ParameterType.TEXT, but the TypeDict hint is {parameter_type_hint}. Change to str."
            case ParameterType.RANGED_INT:
                assert (
                    parameter_type_hint is int
                ), f"For key {key}, the parameter type is ParameterType.RANGED_INT, but the TypeDict hint is {parameter_type_hint}. Change to int."
            case ParameterType.INT:
                assert (
                    parameter_type_hint is int
                ), f"For key {key}, the parameter type is ParameterType.INT, but the TypeDict hint is {parameter_type_hint}. Change to int."
            case _: # pragma: no cover
                assert_never(parameter_type)


def type_hinting_get_sample_payload(hints: Dict[str, Any]) -> RequestBody:
    if not is_typeddict(hints["inputs"]):
        raise BadRequestError(f"Inputs must be a TypedDict")
    if not is_typeddict(hints["parameters"]):
        raise BadRequestError(f"Parameters must be a TypedDict")

    input_type_hints: Mapping[str, BaseModel] = get_type_hints(hints["inputs"])  # this is a typed dict
    parameter_type_hints: Mapping[str, Union[str, int, float]] = get_type_hints(
        hints["parameters"]
    )  # this is a typed dict

    inputs: Dict[str, Input] = {}
    parameters: Dict[str, Union[str, int, float]] = {}

    for key, input_type in input_type_hints.items():
        inputs[key] = resolve_input_sample(input_type)

    for key, parameter_type in parameter_type_hints.items():
        if parameter_type is str:
            parameters[key] = "Sample value for parameter"
        elif parameter_type is int:
            parameters[key] = 1
        elif parameter_type is float:
            parameters[key] = 1.0
        else:
            raise BadRequestError(f"Unsupported parameter type: {parameter_type}")

    return RequestBody(inputs=inputs, parameters=parameters)


def resolve_input_with_data(input_type: Any, data: Dict[str, Any]):
    return input_type(**data)


def no_schema_get_inputs(inputs_typed_dict_hints: Mapping[str, BaseModel], data_: Dict[str, Any]):
    json_inputs = validate_data_is_dict(data_, "inputs")
    input_keys = set(inputs_typed_dict_hints.keys())
    json_keys = set(json_inputs.keys())
    if input_keys != json_keys:
        raise BadRequestError(
            f"Keys mismatch. The input schema has {input_keys=} while your json data has {json_keys=}. Ensure the request body contains all keys under the key 'inputs'. Call /api/routes to see how to use the API."
        )
    return {
        key: resolve_input_with_data(inputs_typed_dict_hints[key], json_inputs[key]) for key in input_keys
    }


def no_schema_get_parameters(
    parameters_typed_dict_hints: Mapping[str, Union[str, int, float]], data_: Dict[str, Any]
) -> Dict[str, Union[str, int, float]]:
    json_parameters = validate_data_is_dict(data_, "parameters")
    parameter_keys = set(parameters_typed_dict_hints.keys())
    json_keys = set(json_parameters.keys())
    if parameter_keys != json_keys:
        raise BadRequestError(
            f"Keys mismatch. The parameter schema has {parameter_keys=} while your json data has {json_keys=}. Ensure the request body contains all keys under the key 'parameters'. Call /api/routes to see how to use the API."
        )
    return {key: json_parameters[key] for key in parameter_keys}
