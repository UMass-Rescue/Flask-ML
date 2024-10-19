from typing import TypedDict, get_type_hints
import pytest
from flask_ml.flask_ml_server.errors import BadRequestError
from flask_ml.flask_ml_server.models import *
from flask_ml.flask_ml_server.utils import (
    validate_data_is_dict,
    validate_data_has_keys,
    input_from_data,
    schema_get_inputs,
    schema_get_parameters,
    schema_get_sample_payload,
    resolve_input_sample,
    ensure_ml_func_parameters_are_typed_dict,
    ensure_ml_func_hinting_and_task_schemas_are_valid,
    type_hinting_get_sample_payload,
    resolve_input_with_data,
    no_schema_get_inputs,
    no_schema_get_parameters
)
from .constants import *

def test_validate_data_is_dict_on_valid_data():
    data = RequestBody(
        inputs={ "text_input": Input(root=TextInput(text="Hello")) },
        parameters={},
    ).model_dump()
    assert validate_data_is_dict(data) == data

def test_validate_data_is_dict_on_invalid_data():
    data = [{"inputs"}]
    with pytest.raises(BadRequestError, match="must be a valid JSON dictionary."):
        validate_data_is_dict(data)

def test_validate_data_has_keys_on_invalid_data():
    data = RequestBody(
        inputs={ "text_input": Input(root=TextInput(text="Hello")) },
        parameters={},
    ).model_dump()
    with pytest.raises(BadRequestError, match="must be a valid JSON dictionary and contain exactly the keys in"):
        validate_data_has_keys(data, ["inputs", "parameters", "extra"])

@pytest.mark.parametrize("input_type, data, model", [
    (InputType.TEXT, {"text": "Hello"}, TextInput),
    (InputType.TEXTAREA, {"text": "Hello"}, TextInput),
    (InputType.BATCHTEXT, {"texts": [{"text": "Hello"}, {"text": "World"}]}, BatchTextInput),
    (InputType.FILE, {"path": "file.txt"}, FileInput),
    (InputType.BATCHFILE, {"files": [{"path": "file1.txt"}, {"path": "file2.txt"}]}, BatchFileInput),
    (InputType.DIRECTORY, {"path": "directory"}, DirectoryInput),
    (InputType.BATCHDIRECTORY, {"directories": [{"path": "dir1"}, {"path": "dir2"}]}, BatchDirectoryInput),
])
def test_input_from_data_on_valid_data(input_type, data, model):
    result = input_from_data(input_type, data)
    assert isinstance(result, model)
    assert result.model_dump() == data

def test_input_from_data_on_invalid_data():
    input_type = "invalid_type"
    data = { "invalid_key": "Hello" }
    with pytest.raises(AssertionError):
        input_from_data(input_type, data) # type: ignore

def test_schema_get_inputs_on_valid_data():
    task_schema = TaskSchema(
        inputs = [InputSchema(
            key = "text_input",
            input_type = InputType.TEXT,
            label = "Text Input",
        )],
        parameters = [],
    )
    inputs = {
        "text_input": { "text": "Hello" }
    }
    result = schema_get_inputs(task_schema, inputs)
    assert result == { "text_input": TextInput(text="Hello") }

def test_schema_get_inputs_on_invalid_data():
    task_schema = TaskSchema(
        inputs = [InputSchema(
            key = "text_input",
            input_type = InputType.TEXT,
            label = "Text Input",
        )],
        parameters = [],
    )
    inputs = {
        "invalid_key": { "text": "Hello" }
    }
    with pytest.raises(BadRequestError, match="Keys mismatch. The input schema has"):
        schema_get_inputs(task_schema, inputs)

def test_schema_get_parameters_on_valid_data():
    task_schema = TaskSchema(
        inputs = [],
        parameters = [ParameterSchema(
            key = "param1",
            label = "Parameter 1",
            value=RangedIntParameterDescriptor(range=IntRangeDescriptor(min=0, max=10), default=5),
        )],
    )
    parameters = {
        "param1": 0
    }
    result = schema_get_parameters(task_schema, parameters)
    assert result == { "param1": 0 }

def test_schema_get_parameters_on_invalid_data():
    task_schema = TaskSchema(
        inputs = [],
        parameters = [ParameterSchema(
            key = "param1",
            label = "Parameter 1",
            value=RangedIntParameterDescriptor(range=IntRangeDescriptor(min=0, max=10), default=5),
        )],
    )
    parameters = {
        "invalid_key": 0
    }
    with pytest.raises(BadRequestError, match="Keys mismatch. The parameter schema has"):
        schema_get_parameters(task_schema, parameters)

@pytest.mark.parametrize("input_schema, expected_inputs", [
    (TEXT_INPUT_SCHEMA, {"text_input": Input(root=TextInput(text="A sample piece of text"))}),
    (TEXTAREA_INPUT_SCHEMA, {"text_input": Input(root=TextInput(text="A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long. A sample piece of text of text that's long."))}),
    (BATCHTEXT_INPUT_SCHEMA, {"text_inputs": Input(root=BatchTextInput(texts=[TextInput(text="A sample piece of text 1"), TextInput(text="A sample piece of text 2")]))}),
    (FILE_INPUT_SCHEMA, {"file_input": Input(root=FileInput(path="/Users/path/to/file"))}),
    (BATCHFILE_INPUT_SCHEMA, {"file_inputs": Input(root=BatchFileInput(files=[FileInput(path="/Users/path/to/file1"), FileInput(path="/Users/path/to/file2")]))}),
    (DIRECTORY_INPUT_SCHEMA, {'dir_input': Input(root=DirectoryInput(path='/Users/path/to/folder'))}),
    (BATCHDIRECTORY_INPUT_SCHEMA, {"dir_inputs": Input(root=BatchDirectoryInput(directories=[DirectoryInput(path="/Users/path/to/folder1"), DirectoryInput(path="/Users/path/to/folder2")]))}),
])
def test_schema_get_sample_payload_on_valid_input(input_schema, expected_inputs):
    task_schema = TaskSchema(inputs=[input_schema], parameters=[])
    result = schema_get_sample_payload(task_schema)
    assert result == RequestBody(inputs=expected_inputs, parameters={})

@pytest.mark.parametrize("parameter_schema, expected_parameters", [
    (TEXT_PARAM_SCHEMA, {"param1": "default"}),
    (ENUM_PARAM_SCHEMA, {"param1": "option_1"}),
    (FLOAT_PARAM_SCHEMA, {"param1": 0.0}),
    (INT_PARAM_SCHEMA, {"param1": 1}),
    (RANGED_FLOAT_PARAM_SCHEMA, {"param1": 0.0}),
    (RANGED_INT_PARAM_SCHEMA, {"param1": 0}),
])
def test_schema_get_sample_payload_on_valid_parameters(parameter_schema, expected_parameters):
    task_schema = TaskSchema(inputs=[], parameters=[parameter_schema])
    result = schema_get_sample_payload(task_schema)
    assert result == RequestBody(inputs={}, parameters=expected_parameters)

@pytest.mark.parametrize("input_type, expected", [
    (FileInput, Input(root=FileInput(path="/Users/path/to/file"))),
    (BatchFileInput, Input(root=BatchFileInput(files=[FileInput(path="/Users/path/to/file1"), FileInput(path="/Users/path/to/file2")]))),
    (TextInput, Input(root=TextInput(text="A sample piece of text"))),
    (BatchTextInput, Input(root=BatchTextInput(texts=[TextInput(text="A sample piece of text 1"), TextInput(text="A sample piece of text 2")]))),
    (DirectoryInput, Input(root=DirectoryInput(path="/Users/path/to/folder"))),
    (BatchDirectoryInput, Input(root=BatchDirectoryInput(directories=[DirectoryInput(path="/Users/path/to/folder1"), DirectoryInput(path="/Users/path/to/folder2")]))),
])
def test_resolve_input_sample_on_valid_data(input_type, expected):
    result = resolve_input_sample(input_type)
    assert isinstance(result, Input)
    assert result == expected

def test_resolve_input_sample_on_invalid_data():
    with pytest.raises(AssertionError):
        resolve_input_sample("invalid_type")

def test_ensure_ml_func_parameters_are_typed_dict_on_valid_data():
    def ml_function(inputs: TextInput, parameters: FloatParameterDescriptor) -> ResponseBody:
        return ResponseBody(TextResponse(value=inputs.text, title="Hello"))

    result = ensure_ml_func_parameters_are_typed_dict(ml_function)
    assert result == None

def test_ensure_ml_func_parameters_are_typed_dict_on_invalid_inputs():
    def ml_function(inputs: str, parameters: FloatParameterDescriptor) -> ResponseBody:
        return ResponseBody(TextResponse(value=inputs, title="Hello"))

    with pytest.raises(BadRequestError, match="Inputs must be a TypedDict"):
        ensure_ml_func_parameters_are_typed_dict(ml_function)

def test_ensure_ml_func_parameters_are_typed_dict_on_invalid_parameters():
    def ml_function(inputs: TextInput, parameters: str) -> ResponseBody:
        return ResponseBody(TextResponse(value=inputs.text, title="Hello"))

    with pytest.raises(BadRequestError, match="Parameters must be a TypedDict"):
        ensure_ml_func_parameters_are_typed_dict(ml_function)

def test_type_hinting_get_sample_payload_on_valid_data():
    class Inputs(TypedDict):
        text_input: TextInput
        file_input: FileInput
        dir_input: DirectoryInput
        batch_text_inputs: BatchTextInput
        batch_file_inputs: BatchFileInput
        batch_directory_inputs: BatchDirectoryInput

    class Parameters(TypedDict):
        param1: float
        param2: int
        param3: str
    
    def ml_function(inputs: Inputs, parameters: Parameters) -> ResponseBody:
        return ResponseBody(TextResponse(value=inputs['text_input'].text, title="Hello"))

    hints = get_type_hints(ml_function)
    result = type_hinting_get_sample_payload(hints)
    assert isinstance(result, RequestBody)

def test_type_hinting_get_sample_payload_on_invalid_inputs():
    class Parameters(TypedDict):
        param1: float
    
    def ml_function(inputs: List[str], parameters: Parameters) -> ResponseBody:
        return ResponseBody(TextResponse(value="Hello", title="Hello"))

    hints = get_type_hints(ml_function)
    with pytest.raises(BadRequestError, match="Inputs must be a TypedDict"):
        type_hinting_get_sample_payload(hints) 

def test_type_hinting_get_sample_payload_on_invalid_parameters():
    class Inputs(TypedDict):
        text_input: TextInput
    
    def ml_function(inputs: Inputs, parameters: List[str]) -> ResponseBody:
        return ResponseBody(TextResponse(value="Hello", title="Hello"))

    hints = get_type_hints(ml_function)
    with pytest.raises(BadRequestError, match="Parameters must be a TypedDict"):
        type_hinting_get_sample_payload(hints) 

def test_type_hinting_get_sample_payload_on_invalid_parameter_type():
    class Inputs(TypedDict):
        text_input: TextInput
    
    class Parameters(TypedDict):
        param1: List[str]
    
    def ml_function(inputs: Inputs, parameters: Parameters) -> ResponseBody:
        return ResponseBody(TextResponse(value="Hello", title="Hello"))

    hints = get_type_hints(ml_function)
    with pytest.raises(BadRequestError, match="Unsupported parameter type"):
        type_hinting_get_sample_payload(hints) 

@pytest.mark.parametrize("input_type, data", [
    (TextInput, {"text": "Hello"}),
    (BatchTextInput, {"texts": [{"text": "Hello"}, {"text": "World"}]}),
    (FileInput, {"path": "file.txt"}),
    (BatchFileInput, {"files": [{"path": "file1.txt"}, {"path": "file2.txt"}]}),
    (DirectoryInput, {"path": "directory"}),
    (BatchDirectoryInput, {"directories": [{"path": "dir1"}, {"path": "dir2"}]}),
])
def test_resolve_input_with_data_on_valid_data(input_type, data):
    result = resolve_input_with_data(input_type, data)
    assert isinstance(result, input_type)
    assert result.model_dump() == data

def test_no_schema_get_inputs_on_valid_data():
    class Inputs(TypedDict):
        text_input: TextInput
        file_input: FileInput

    hints = get_type_hints(Inputs)
    inputs = {
        "text_input": {"text": "Hello"},
        "file_input": {"path": "file.txt"},
    }
    result = no_schema_get_inputs(hints, inputs)
    assert result == {"text_input": TextInput(text="Hello"), "file_input": FileInput(path="file.txt")}

def test_no_schema_get_inputs_on_invalid_data():
    class Inputs(TypedDict):
        text_input: TextInput
        file_input: FileInput

    hints = get_type_hints(Inputs)
    inputs = {
        "invalid_key": {"text": "Hello"},
    }
    with pytest.raises(BadRequestError, match="Keys mismatch. The input schema has"):
        no_schema_get_inputs(hints, inputs)

def test_no_schema_get_parameters_on_valid_data():
    class Parameters(TypedDict):
        param1: int
        param2: str

    hints = get_type_hints(Parameters)
    parameters = {
        "param1": 1,
        "param2": "test",
    }
    result = no_schema_get_parameters(hints, parameters)
    assert result == {"param1": 1, "param2": "test"}

def test_no_schema_get_parameters_on_invalid_data():
    class Parameters(TypedDict):
        param1: int
        param2: str

    hints = get_type_hints(Parameters)
    parameters = {
        "invalid_key": 1,
    }
    with pytest.raises(BadRequestError, match="Keys mismatch. The parameter schema has"):
        no_schema_get_parameters(hints, parameters)  # type: ignore

def test_ensure_ml_func_hinting_and_task_schemas_are_valid_on_valid_data():
    class Inputs(TypedDict):
        text_input: TextInput
        textarea_input: TextInput
        file_input: FileInput
        dir_input: DirectoryInput
        batch_text_inputs: BatchTextInput
        batch_file_inputs: BatchFileInput
        batch_directory_inputs: BatchDirectoryInput

    class Parameters(TypedDict):
        param1: float
        param2: int
        param3: str
        param4: str
        param5: float
        param6: int
    
    def ml_function(inputs: Inputs, parameters: Parameters) -> ResponseBody:
        return ResponseBody(TextResponse(value=inputs['text_input'].text, title="Hello"))

    task_schema = TaskSchema(
        inputs=[
            InputSchema(key="text_input", input_type=InputType.TEXT, label="Text Input"),
            InputSchema(key="textarea_input", input_type=InputType.TEXTAREA, label="Textarea Input"),
            InputSchema(key="file_input", input_type=InputType.FILE, label="File Input"),
            InputSchema(key="dir_input", input_type=InputType.DIRECTORY, label="Directory Input"),
            InputSchema(key="batch_text_inputs", input_type=InputType.BATCHTEXT, label="Batch Text Inputs"),
            InputSchema(key="batch_file_inputs", input_type=InputType.BATCHFILE, label="Batch File Inputs"),
            InputSchema(key="batch_directory_inputs", input_type=InputType.BATCHDIRECTORY, label="Batch Directory Inputs"),
        ],
        parameters=[
            ParameterSchema(key="param1", label="Parameter 1", value=FloatParameterDescriptor(parameter_type=ParameterType.FLOAT, default=0.0)),
            ParameterSchema(key="param2", label="Parameter 2", value=IntParameterDescriptor(parameter_type=ParameterType.INT, default=5)),
            ParameterSchema(key="param3", label="Parameter 3", value=TextParameterDescriptor(parameter_type=ParameterType.TEXT, default="default")),
            ParameterSchema(key="param4", label="Parameter 4", value=EnumParameterDescriptor(parameter_type=ParameterType.ENUM, enum_vals=[EnumVal(label="Option 1", key="option_1"), EnumVal(label="Option 2", key="option_2")], default="option_1")),
            ParameterSchema(key="param5", label="Parameter 5", value=RangedFloatParameterDescriptor(parameter_type=ParameterType.RANGED_FLOAT, range=FloatRangeDescriptor(min=0.0, max=1.0) , default=0.0)),
            ParameterSchema(key="param6", label="Parameter 6", value=RangedIntParameterDescriptor(parameter_type=ParameterType.RANGED_INT, range=IntRangeDescriptor(min=1, max=10), default=5)),
        ]
    )

    result = ensure_ml_func_hinting_and_task_schemas_are_valid(ml_function, task_schema)
    assert result is None
