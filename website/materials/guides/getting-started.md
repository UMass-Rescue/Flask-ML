---
sidebar_position: 1
---

# Getting Started with Flask-ML

## Introduction
Flask-ML is a Flask extension that allows you to run machine learning code in a Flask server. In this tutorial, we will walk you through the entire process of creating a Flask-ML server to expose your machine learning model, as well as defining a UI schema for your ML task so that it will have a nice user interface to go along with it.

## Features
- Implement and run a Flask server exposing standard endpoints to run your machine learning code
- Write a schema that lets UI clients automatically generate a user interface for your machine learning code

## Objectives
By the end of this tutorial, you will be able to:

- Create a Flask-ML server
- Define inputs and outputs for your machine learning code
- Write type-safe Python code
- Expose a UI schema for your machine learning code

## Tutorial

### Creating a Project and Installing Flask-ML
To get started, create a new directory for your project and navigate to it in your terminal. Then, run the following command to create a new Python project:

```bash
python -m venv venv
source venv/bin/activate
pip install Flask-ML
```

This will create a new virtual environment and install Flask-ML.

### Adding a `pyrightconfig.json` File

If you use VSCode, make sure you have [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) installed. Then, add a `pyrightconfig.json` file to your project directory. Adding this activates type-checking for this Python project, helping you write type-safe code. This is important, so please do not skip this step.

```json
{
    "python.analysis.typeCheckingMode": "basic"
}
```

### Creating a `server.py` File
Create a new file called `server.py` in your project directory. This file will contain the code for your Flask-ML server.

```python
from flask_ml.flask_ml_server import MLServer

server = MLServer(__name__)

# Run a debug flask server
server.run()
```

### Writing a Basic Inference Function
Now, let's write a basic inference function. For example, let's say that our function takes in a collection of raw text inputs, transforms then into either lower or upper case, and returns a collection of processed text outputs.

```python
@server.route("/transform_case")
def transform_case(inputs, parameters):
    pass
```

In Flask-ML, an inference function takes two arguments: `inputs` and `parameters`. We also add the decorator to register our function with Flask-ML at the endpoint `/transform_case`.

### Adding Types for Inputs and Parameters
Let's add types to our inputs and parameters. In Flask-ML, the types of inputs and parameters must be Python [TypedDict](https://docs.python.org/3/library/typing.html#typing.TypedDict) types.

Flask-ML offers the following types of inputs for the keys within the TypedDict:

- `TextInput`: a single raw text input
- `FileInput`: a single file path
- `DirectoryInput`: a directory path
- `BatchTextInput`: a collection of raw text inputs
- `BatchFileInput`: a collection of file paths
- `BatchDirectoryInput`: a collection of directory paths

The type of parameters must be either `str`, `int`, `float`, or `bool`.

```python
from typing import TypedDict
from flask_ml.flask_ml_server.models import BatchTextInput

class TransformCaseInputs(TypedDict):
    text_inputs: BatchTextInput

class TransformCaseParameters(TypedDict):
    to_case: str # 'upper' or 'lower'

@server.route("/transform_case")
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters):
    pass
```

### Writing a Return Type
Now, let's add a return type. Flask-ML offers the following return types:

- `TextResponse`: a single raw text output
- `FileResponse`: a file path
- `DirectoryResponse`: a directory path
- `MarkdownResponse`: a markdown string
- `BatchTextResponse`: a collection of raw text outputs
- `BatchFileResponse`: a collection of file paths
- `BatchDirectoryResponse`: a collection of directory paths

```python
from typing import TypedDict
from flask_ml.flask_ml_server.models import BatchTextInput, ResponseBody, BatchTextResponse

class TransformCaseInputs(TypedDict):
    text_inputs: BatchTextInput

class TransformCaseParameters(TypedDict):
    to_case: str # 'upper' or 'lower'

@server.route("/transform_case")
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    return ResponseBody(root=BatchTextResponse(texts=[]))
```

Since we want to return a collection of raw text outputs, we will use the `BatchTextResponse` type. Note that the return type of the function is `ResponseBody`, which is a generic type that can be used to represent any type of response. We will add the `BatchTextResponse` in it.

### Implementing the Model
Now, let's implement the model:

```python
from typing import TypedDict
from flask_ml.flask_ml_server.models import BatchTextInput, ResponseBody, BatchTextResponse, TextResponse

class TransformCaseInputs(TypedDict):
    text_inputs: BatchTextInput

class TransformCaseParameters(TypedDict):
    to_case: str # 'upper' or 'lower'

@server.route("/transform_case")
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    to_upper: bool = parameters['to_case'] == 'upper'
    
    outputs = []
    for text_input in inputs['text_inputs'].texts:
        raw_text = text_input.text
        processed_text = raw_text.upper() if to_upper else raw_text.lower()
        outputs.append(TextResponse(value=processed_text, title=raw_text))

    return ResponseBody(root=BatchTextResponse(texts=outputs))
```

Here, we use the `to_case` parameter to determine whether to transform the text to upper case or lower case. We then iterate over the `text_inputs` and transform each input using the appropriate case transformation. Finally, we return a `BatchTextResponse` containing the transformed text outputs. 

### First Run
We are ready to run our first task! Note that we haven't defined a UI schema yet, so we don't have access to those features yet. However, let's check out what we can do so far.

Run the server with:

```bash
python server.py
```

Open Postman or an HTTP client of your choice. Make a GET request to `http://localhost:5000/api/routes` as follows:

```bash
curl --location 'http://localhost:5000/api/routes'
```

which will produce the following JSON response:

```json
[
    {
        "payload_schema": "/transform_case/payload_schema",
        "run_task": "/transform_case",
        "sample_payload": "/transform_case/sample_payload"
    }
]
```

This indicates that the server has registered our route at `/transform_case`. You can ignore `payload_schema`, although you can simply try calling it to see what it returns. Let's call `/transform_case/sample_payload` to get an idea of what the expected request body looks like:

```bash
curl --location 'http://localhost:5000/transform_case/sample_payload'
```

which returns:

```json
{
    "inputs": {
        "text_inputs": {
            "texts": [
                {
                    "text": "A sample piece of text 1"
                },
                {
                    "text": "A sample piece of text 2"
                }
            ]
        }
    },
    "parameters": {
        "to_case": "Sample value for parameter"
    }
}
```

Note that here, for the `to_case` parameter above, the sample value isn't valid because we need it to be one of `"upper"` or `"lower"`, so make sure that when you go through the next step, you change it appropriately.

### Running the `transform_case` Task

Now, let's make a POST request to `/transform_case` with the following JSON payload. Make sure you set the `to_case` parameter to `"upper"` or `"lower"` depending on what you want to transform the text to.

```bash
curl --location 
    -X POST
    -H 'Content-Type: application/json'
    -d '{
        "inputs": {
            "text_inputs": {
                "texts": [
                    {
                        "text": "A sample piece of text 1"
                    },
                    {
                        "text": "A sample piece of text 2"
                    }
                ]
            }
        },
        "parameters": {
            "to_case": "upper"
        }
    }'
    'http://localhost:5000/transform_case'
```

which produces the following response:

```json
{
    "output_type": "batchtext",
    "texts": [
        {
            "output_type": "text",
            "value": "A SAMPLE PIECE OF TEXT 1",
            "title": "a sample piece of text 1",
            "subtitle": null
        },
        {
            "output_type": "text",
            "value": "A SAMPLE PIECE OF TEXT 2",
            "title": "a sample piece of text 2",
            "subtitle": null
        }
    ]
}
```

Great! We got our first task running!

### Adding a UI Schema
To define a UI for this sample model, we need to write a task schema function. This function will return a `TaskSchema` object, which contains specifications for the types and possible values for all the inputs and parameters for our ML function.

First, we will write a schema for our inputs by building an `InputSchema`. We simply specify a `key` (which must match the key for the respective attribute in the `TypedDict` defined for our inputs), a `label` to be displayed on the UI, and the `input_type`, which must be one of the following:

- `InputType.TEXT`: a single raw text input
- `InputType.FILE`: a single file path
- `InputType.DIRECTORY`: a single directory path
- `InputType.MARKDOWN`: a markdown string
- `InputType.BATCHTEXT`: a collection of raw text inputs
- `InputType.BATCHFILE`: a collection of file paths
- `InputType.BATCHDIRECTORY`: a collection of directory paths

Let's write an input schema for our function's inputs:

```python
inputSchema = InputSchema(
    key="text_inputs",
    label="Text to Transform",
    input_type=InputType.BATCHTEXT
)
```

Next, we will write a similar schema for our parameters by building an `ParameterSchema`. We again specify a `key` (which must match the key for the respective attribute in the `TypedDict` defined for our parameters), a `label` to be displayed on the UI, an optional `subtitle` that is more expansive than the label, and the `value`, which must be an instance of one of these types:

- `TextParameterDescriptor`: a string of raw text
    - `default`: default value
- `EnumParameterDescriptor`: an enumeration with multiple values
    - `enum_vals`: list of `EnumVal` objects, each containing a unique `key` and `label`
    - `message_when_empty`: message to display on UI if this enum contains no values (optional)
    - `default`: default value
- `IntParameterDescriptor`
    - `default`: default value
- `RangedIntParameterDescriptor`
    - `default`: default value
    - `range`: an `IntRangeDescriptor` containing a `min` value and `max` value
- `FloatParameterDescriptor`
    - `default`: default value
- `RangedFloatParameterDescriptor`
    - `default`: default value
    - `range`: a `FloatRangeDescriptor` containing a `min` value and `max` value

Let's write a parameter schema for our function's parameters:

```python
parameterSchema = ParameterSchema(
    key="to_case",
    label="Case to Transform Text Into",
    subtitle="'upper' will convert all text to upper case. 'lower' will convert all text to lower case.",
    value=EnumParameterDescriptor(
        enum_vals=[
            EnumVal(
                key="upper",
                label="UPPER"
            ),
            EnumVal(
                key="lower",
                label="LOWER"
            )
        ],
        default="upper"
    )
)
```

Let's use the above schemas to define a UI schema for our `transform_case` task:

```python
from flask_ml.flask_ml_server.models import TaskSchema, InputSchema, ParameterSchema, InputType, EnumParameterDescriptor, EnumVal

def create_transform_case_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs = [inputSchema],
        parameters = [parameterSchema]
    )
```

Now, we can register this function in our endpoint by setting the `task_schema_func` parameter in the route annotation:

```python
@server.route("/transform_case", task_schema_func=create_transform_case_task_schema)
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    ...
```

We can also set a few other parameters within the annotation in order to provide the UI some extra information:
- `short_title`: concise name for your task, to be displayed as a heading in the UI
- `order`: if you support multiple ML tasks (endpoints), this defines the order in which the task tabs should appear in the UI (starts from 0) 

```python
@server.route("/transform_case", task_schema_func=create_transform_case_task_schema, short_title="Transform Case", order=0)
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    ...
```

Note that if you call `/api/routes` again, you will get a more extensive response this time:

```json
[
    {
        "order": 0,
        "payload_schema": "/transform_case/payload_schema",
        "run_task": "/transform_case",
        "sample_payload": "/transform_case/sample_payload",
        "short_title": "Transform Case",
        "task_schema": "/transform_case/task_schema"
    }
]
```

Note that calling the `/transform_case/task_schema` route simply executes the TaskSchema function that we had written and returns its output as a response. This contains all the information that the UI will need in order to determine how to render each of the inputs and parameters in its forms.

## Entire Sample Code

```python
from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import BatchTextInput, BatchTextResponse, EnumParameterDescriptor, EnumVal, InputSchema, InputType, ParameterSchema, ResponseBody, TaskSchema, TextResponse

server = MLServer(__name__)

class TransformCaseInputs(TypedDict):
    text_inputs: BatchTextInput

class TransformCaseParameters(TypedDict):
    to_case: str # 'upper' or 'lower'

def create_transform_case_task_schema() -> TaskSchema:
    inputSchema = InputSchema(
        key="text_inputs",
        label="Text to Transform",
        input_type=InputType.BATCHTEXT
    )
    parameterSchema = ParameterSchema(
        key="to_case",
        label="Case to Transform Text Into",
        subtitle="'upper' will convert all text to upper case. 'lower' will convert all text to lower case.",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(
                    key="upper",
                    label="UPPER"
                ),
                EnumVal(
                    key="lower",
                    label="LOWER"
                )
            ],
            default="upper"
        )
    )
    return TaskSchema(
        inputs = [inputSchema],
        parameters = [parameterSchema]
    )

@server.route(
    "/transform_case",
    task_schema_func=create_transform_case_task_schema,
    short_title="Transform Case",
    order=0
)
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    to_upper: bool = parameters['to_case'] == 'upper'
    
    outputs = []
    for text_input in inputs['text_inputs'].texts:
        raw_text = text_input.text
        processed_text = raw_text.upper() if to_upper else raw_text.lower()
        outputs.append(TextResponse(value=processed_text, title=raw_text))

    return ResponseBody(root=BatchTextResponse(texts=outputs))

if __name__ == "__main__":
    # Run a debug server
    server.run()

```