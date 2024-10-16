from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Union,
    assert_never,
    get_type_hints,
)

from pydantic import BaseModel

from flask_ml.flask_ml_server.errors import BadRequestError
from flask_ml.flask_ml_server.MLServer import TaskSchema
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    BatchTextInput,
    DirectoryInput,
    EnumParameterDescriptor,
    FileInput,
    FloatParameterDescriptor,
    Input,
    InputType,
    IntParameterDescriptor,
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
    if not set(data.keys()).issubset(set(keys)):
        raise BadRequestError(
            f"Request body must be a valid JSON dictionary and contain all keys in {keys=}. Provided request body {data_} is not a valid. Call /api/routes to see how to use the API."
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
        case _:
            assert_never(input_type)


def schema_get_inputs(schema: TaskSchema, data_: Dict[str, Any]):
    json_inputs = validate_data_is_dict(data_, "inputs")
    input_schema = schema.inputs
    input_keys_to_input_type = {inputt.key: inputt.input_type for inputt in input_schema}
    input_keys = list(input_keys_to_input_type.keys())
    json_keys = list(json_inputs.keys())
    if input_keys != json_keys:
        raise BadRequestError(
            f"Keys mismatch. The input schema has {input_keys=} while your json data has {json_keys=}. Ensure the request body contains all keys under the key 'inputs'. Call /api/routes to see how to use the API."
        )
    return {key: input_from_data(input_keys_to_input_type[key], json_inputs[key]) for key in input_keys}


def schema_get_parameters(schema: TaskSchema, data_: Dict[str, Any]) -> Dict[str, Union[str, int, float]]:
    json_parameters = validate_data_is_dict(data_, "parameters")
    parameter_schema = schema.parameters
    parameter_keys = [parameter.key for parameter in parameter_schema]
    json_keys = list(json_parameters.keys())
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
            case _:
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
            case _:
                assert_never(parameter_schema.value)
    return RequestBody(inputs=inputs, parameters=parameters)


def resolve_input_sample(input_type: Any) -> Input:
    if input_type is FileInput:
        return Input(root=FileInput(path="/Users/path/to/file"))
    elif input_type is DirectoryInput:
        return Input(root=DirectoryInput(path="/Users/path/to/folder"))
    elif input_type is TextInput:
        return Input(root=TextInput(text="A sample piece of text"))
    elif input_type is BatchFileInput:
        return Input(
            root=BatchFileInput(
                files=[
                    FileInput(path="/Users/path/to/file1"),
                    FileInput(path="/Users/path/to/file2"),
                ]
            )
        )
    elif input_type is BatchTextInput:
        return Input(
            root=BatchTextInput(
                texts=[
                    TextInput(text="A sample piece of text 1"),
                    TextInput(text="A sample piece of text 2"),
                ]
            )
        )
    else:
        raise BadRequestError(f"Unsupported input type: {input_type}")


def is_typeddict(cls):
    return isinstance(cls, type) and hasattr(cls, "__annotations__")


def ensure_ml_func_parameters_are_typed_dict(ml_function: Callable[[Any, Any], ResponseBody]):
    hints = get_type_hints(ml_function)
    if not is_typeddict(hints["inputs"]):
        raise BadRequestError(f"Inputs must be a TypedDict")
    if not is_typeddict(hints["parameters"]):
        raise BadRequestError(f"Parameters must be a TypedDict")


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
    if input_type is FileInput:
        return FileInput(**data)
    elif input_type is DirectoryInput:
        return DirectoryInput(**data)
    elif input_type is TextInput:
        return TextInput(**data)
    elif input_type is BatchFileInput:
        return BatchFileInput(
            **data,
        )
    elif input_type is BatchTextInput:
        return BatchTextInput(
            **data,
        )
    else:
        raise BadRequestError(f"Unsupported input type: {input_type}")


def no_schema_get_inputs(inputs_typed_dict_hints: Mapping[str, BaseModel], data_: Dict[str, Any]):
    json_inputs = validate_data_is_dict(data_, "inputs")
    input_keys = list(inputs_typed_dict_hints.keys())
    json_keys = list(json_inputs.keys())
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
    parameter_keys = list(parameters_typed_dict_hints.keys())
    json_keys = list(json_parameters.keys())
    if parameter_keys != json_keys:
        raise BadRequestError(
            f"Keys mismatch. The parameter schema has {parameter_keys=} while your json data has {json_keys=}. Ensure the request body contains all keys under the key 'parameters'. Call /api/routes to see how to use the API."
        )
    return {key: json_parameters[key] for key in parameter_keys}
