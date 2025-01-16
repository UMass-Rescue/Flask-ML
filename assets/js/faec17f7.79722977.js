"use strict";(self.webpackChunkFlask_ML=self.webpackChunkFlask_ML||[]).push([[141],{2312:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>o,default:()=>h,frontMatter:()=>i,metadata:()=>s,toc:()=>c});const s=JSON.parse('{"id":"guides/getting-started","title":"Getting Started with Flask-ML","description":"Introduction","source":"@site/materials/guides/getting-started.md","sourceDirName":"guides","slug":"/guides/getting-started","permalink":"/Flask-ML/materials/guides/getting-started","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","previous":{"title":"Guides","permalink":"/Flask-ML/materials/category/guides"},"next":{"title":"Writing a CLI","permalink":"/Flask-ML/materials/guides/cli"}}');var a=t(4848),r=t(8453);const i={sidebar_position:1},o="Getting Started with Flask-ML",l={},c=[{value:"Introduction",id:"introduction",level:2},{value:"Features",id:"features",level:2},{value:"Objectives",id:"objectives",level:2},{value:"Tutorial",id:"tutorial",level:2},{value:"Creating a Project and Installing Flask-ML",id:"creating-a-project-and-installing-flask-ml",level:3},{value:"Adding a <code>pyrightconfig.json</code> File",id:"adding-a-pyrightconfigjson-file",level:3},{value:"Creating a <code>server.py</code> File",id:"creating-a-serverpy-file",level:3},{value:"Writing a Basic Inference Function",id:"writing-a-basic-inference-function",level:3},{value:"Adding Types for Inputs and Parameters",id:"adding-types-for-inputs-and-parameters",level:3},{value:"Writing a Return Type",id:"writing-a-return-type",level:3},{value:"Implementing the Model",id:"implementing-the-model",level:3},{value:"First Run",id:"first-run",level:3},{value:"Running the <code>transform_case</code> Task",id:"running-the-transform_case-task",level:3},{value:"Adding a UI Schema",id:"adding-a-ui-schema",level:3},{value:"Entire Sample Code",id:"entire-sample-code",level:2},{value:"Adding a automatically generated CLI",id:"adding-a-automatically-generated-cli",level:2},{value:"Adding Application Metadata",id:"adding-application-metadata",level:2}];function d(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",p:"p",pre:"pre",ul:"ul",...(0,r.R)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(n.header,{children:(0,a.jsx)(n.h1,{id:"getting-started-with-flask-ml",children:"Getting Started with Flask-ML"})}),"\n",(0,a.jsx)(n.h2,{id:"introduction",children:"Introduction"}),"\n",(0,a.jsx)(n.p,{children:"Flask-ML is a Flask extension that allows you to run machine learning code in a Flask server. In this tutorial, we will walk you through the entire process of creating a Flask-ML server to expose your machine learning model, as well as defining a UI schema for your ML task so that it will have a nice user interface to go along with it."}),"\n",(0,a.jsx)(n.h2,{id:"features",children:"Features"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsx)(n.li,{children:"Implement and run a Flask server exposing standard endpoints to run your machine learning code"}),"\n",(0,a.jsx)(n.li,{children:"Write a schema that lets UI clients automatically generate a user interface for your machine learning code"}),"\n"]}),"\n",(0,a.jsx)(n.h2,{id:"objectives",children:"Objectives"}),"\n",(0,a.jsx)(n.p,{children:"By the end of this tutorial, you will be able to:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsx)(n.li,{children:"Create a Flask-ML server"}),"\n",(0,a.jsx)(n.li,{children:"Define inputs and outputs for your machine learning code"}),"\n",(0,a.jsx)(n.li,{children:"Write type-safe Python code"}),"\n",(0,a.jsx)(n.li,{children:"Expose a UI schema for your machine learning code"}),"\n"]}),"\n",(0,a.jsx)(n.h2,{id:"tutorial",children:"Tutorial"}),"\n",(0,a.jsx)(n.h3,{id:"creating-a-project-and-installing-flask-ml",children:"Creating a Project and Installing Flask-ML"}),"\n",(0,a.jsx)(n.p,{children:"To get started, create a new directory for your project and navigate to it in your terminal. Then, run the following command to create a new Python project:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"python -m venv venv\nsource venv/bin/activate\npip install Flask-ML\n"})}),"\n",(0,a.jsx)(n.p,{children:"This will create a new virtual environment and install Flask-ML."}),"\n",(0,a.jsxs)(n.h3,{id:"adding-a-pyrightconfigjson-file",children:["Adding a ",(0,a.jsx)(n.code,{children:"pyrightconfig.json"})," File"]}),"\n",(0,a.jsxs)(n.p,{children:["If you use VSCode, make sure you have ",(0,a.jsx)(n.a,{href:"https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance",children:"Pylance"})," installed. Then, add a ",(0,a.jsx)(n.code,{children:"pyrightconfig.json"})," file to your project directory. Adding this activates type-checking for this Python project, helping you write type-safe code. This is important, so please do not skip this step."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-json",children:'{\n    "python.analysis.typeCheckingMode": "basic"\n}\n'})}),"\n",(0,a.jsxs)(n.h3,{id:"creating-a-serverpy-file",children:["Creating a ",(0,a.jsx)(n.code,{children:"server.py"})," File"]}),"\n",(0,a.jsxs)(n.p,{children:["Create a new file called ",(0,a.jsx)(n.code,{children:"server.py"})," in your project directory. This file will contain the code for your Flask-ML server."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from flask_ml.flask_ml_server import MLServer\n\nserver = MLServer(__name__)\n\n# Run a debug flask server\nserver.run()\n"})}),"\n",(0,a.jsx)(n.h3,{id:"writing-a-basic-inference-function",children:"Writing a Basic Inference Function"}),"\n",(0,a.jsx)(n.p,{children:"Now, let's write a basic inference function. For example, let's say that our function takes in a collection of raw text inputs, transforms then into either lower or upper case, and returns a collection of processed text outputs."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'@server.route("/transform_case")\ndef transform_case(inputs, parameters):\n    pass\n'})}),"\n",(0,a.jsxs)(n.p,{children:["In Flask-ML, an inference function takes two arguments: ",(0,a.jsx)(n.code,{children:"inputs"})," and ",(0,a.jsx)(n.code,{children:"parameters"}),". We also add the decorator to register our function with Flask-ML at the endpoint ",(0,a.jsx)(n.code,{children:"/transform_case"}),"."]}),"\n",(0,a.jsx)(n.h3,{id:"adding-types-for-inputs-and-parameters",children:"Adding Types for Inputs and Parameters"}),"\n",(0,a.jsxs)(n.p,{children:["Let's add types to our inputs and parameters. In Flask-ML, the types of inputs and parameters must be Python ",(0,a.jsx)(n.a,{href:"https://docs.python.org/3/library/typing.html#typing.TypedDict",children:"TypedDict"})," types."]}),"\n",(0,a.jsx)(n.p,{children:"Flask-ML offers the following types of inputs for the keys within the TypedDict:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"TextInput"}),": a single raw text input"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"FileInput"}),": a single file path"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"DirectoryInput"}),": a directory path"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"BatchTextInput"}),": a collection of raw text inputs"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"BatchFileInput"}),": a collection of file paths"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"BatchDirectoryInput"}),": a collection of directory paths"]}),"\n"]}),"\n",(0,a.jsxs)(n.p,{children:["The type of parameters must be either ",(0,a.jsx)(n.code,{children:"str"}),", ",(0,a.jsx)(n.code,{children:"int"}),", ",(0,a.jsx)(n.code,{children:"float"}),", or ",(0,a.jsx)(n.code,{children:"bool"}),"."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from typing import TypedDict\nfrom flask_ml.flask_ml_server.models import BatchTextInput\n\nclass TransformCaseInputs(TypedDict):\n    text_inputs: BatchTextInput\n\nclass TransformCaseParameters(TypedDict):\n    to_case: str # 'upper' or 'lower'\n\n@server.route(\"/transform_case\")\ndef transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters):\n    pass\n"})}),"\n",(0,a.jsx)(n.h3,{id:"writing-a-return-type",children:"Writing a Return Type"}),"\n",(0,a.jsx)(n.p,{children:"Now, let's add a return type. Flask-ML offers the following return types:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"TextResponse"}),": a single raw text output"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"FileResponse"}),": a file path"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"DirectoryResponse"}),": a directory path"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"MarkdownResponse"}),": a markdown string"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"BatchTextResponse"}),": a collection of raw text outputs"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"BatchFileResponse"}),": a collection of file paths"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"BatchDirectoryResponse"}),": a collection of directory paths"]}),"\n"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from typing import TypedDict\nfrom flask_ml.flask_ml_server.models import BatchTextInput, ResponseBody, BatchTextResponse\n\nclass TransformCaseInputs(TypedDict):\n    text_inputs: BatchTextInput\n\nclass TransformCaseParameters(TypedDict):\n    to_case: str # 'upper' or 'lower'\n\n@server.route(\"/transform_case\")\ndef transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:\n    return ResponseBody(root=BatchTextResponse(texts=[]))\n"})}),"\n",(0,a.jsxs)(n.p,{children:["Since we want to return a collection of raw text outputs, we will use the ",(0,a.jsx)(n.code,{children:"BatchTextResponse"})," type. Note that the return type of the function is ",(0,a.jsx)(n.code,{children:"ResponseBody"}),", which is a generic type that can be used to represent any type of response. We will add the ",(0,a.jsx)(n.code,{children:"BatchTextResponse"})," in it."]}),"\n",(0,a.jsx)(n.h3,{id:"implementing-the-model",children:"Implementing the Model"}),"\n",(0,a.jsx)(n.p,{children:"Now, let's implement the model:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from typing import TypedDict\nfrom flask_ml.flask_ml_server.models import BatchTextInput, ResponseBody, BatchTextResponse, TextResponse\n\nclass TransformCaseInputs(TypedDict):\n    text_inputs: BatchTextInput\n\nclass TransformCaseParameters(TypedDict):\n    to_case: str # 'upper' or 'lower'\n\n@server.route(\"/transform_case\")\ndef transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:\n    to_upper: bool = parameters['to_case'] == 'upper'\n    \n    outputs = []\n    for text_input in inputs['text_inputs'].texts:\n        raw_text = text_input.text\n        processed_text = raw_text.upper() if to_upper else raw_text.lower()\n        outputs.append(TextResponse(value=processed_text, title=raw_text))\n\n    return ResponseBody(root=BatchTextResponse(texts=outputs))\n"})}),"\n",(0,a.jsxs)(n.p,{children:["Here, we use the ",(0,a.jsx)(n.code,{children:"to_case"})," parameter to determine whether to transform the text to upper case or lower case. We then iterate over the ",(0,a.jsx)(n.code,{children:"text_inputs"})," and transform each input using the appropriate case transformation. Finally, we return a ",(0,a.jsx)(n.code,{children:"BatchTextResponse"})," containing the transformed text outputs."]}),"\n",(0,a.jsx)(n.h3,{id:"first-run",children:"First Run"}),"\n",(0,a.jsx)(n.p,{children:"We are ready to run our first task! Note that we haven't defined a UI schema yet, so we don't have access to those features yet. However, let's check out what we can do so far."}),"\n",(0,a.jsx)(n.p,{children:"Run the server with:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"python server.py\n"})}),"\n",(0,a.jsxs)(n.p,{children:["Open Postman or an HTTP client of your choice. Make a GET request to ",(0,a.jsx)(n.code,{children:"http://localhost:5000/api/routes"})," as follows:"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"curl --location 'http://localhost:5000/api/routes'\n"})}),"\n",(0,a.jsx)(n.p,{children:"which will produce the following JSON response:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-json",children:'[\n    {\n        "payload_schema": "/transform_case/payload_schema",\n        "run_task": "/transform_case",\n        "sample_payload": "/transform_case/sample_payload"\n    }\n]\n'})}),"\n",(0,a.jsxs)(n.p,{children:["This indicates that the server has registered our route at ",(0,a.jsx)(n.code,{children:"/transform_case"}),". You can ignore ",(0,a.jsx)(n.code,{children:"payload_schema"}),", although you can simply try calling it to see what it returns. Let's call ",(0,a.jsx)(n.code,{children:"/transform_case/sample_payload"})," to get an idea of what the expected request body looks like:"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"curl --location 'http://localhost:5000/transform_case/sample_payload'\n"})}),"\n",(0,a.jsx)(n.p,{children:"which returns:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-json",children:'{\n    "inputs": {\n        "text_inputs": {\n            "texts": [\n                {\n                    "text": "A sample piece of text 1"\n                },\n                {\n                    "text": "A sample piece of text 2"\n                }\n            ]\n        }\n    },\n    "parameters": {\n        "to_case": "Sample value for parameter"\n    }\n}\n'})}),"\n",(0,a.jsxs)(n.p,{children:["Note that here, for the ",(0,a.jsx)(n.code,{children:"to_case"})," parameter above, the sample value isn't valid because we need it to be one of ",(0,a.jsx)(n.code,{children:'"upper"'})," or ",(0,a.jsx)(n.code,{children:'"lower"'}),", so make sure that when you go through the next step, you change it appropriately."]}),"\n",(0,a.jsxs)(n.h3,{id:"running-the-transform_case-task",children:["Running the ",(0,a.jsx)(n.code,{children:"transform_case"})," Task"]}),"\n",(0,a.jsxs)(n.p,{children:["Now, let's make a POST request to ",(0,a.jsx)(n.code,{children:"/transform_case"})," with the following JSON payload. Make sure you set the ",(0,a.jsx)(n.code,{children:"to_case"})," parameter to ",(0,a.jsx)(n.code,{children:'"upper"'})," or ",(0,a.jsx)(n.code,{children:'"lower"'})," depending on what you want to transform the text to."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:'curl --location \n    -X POST\n    -H \'Content-Type: application/json\'\n    -d \'{\n        "inputs": {\n            "text_inputs": {\n                "texts": [\n                    {\n                        "text": "A sample piece of text 1"\n                    },\n                    {\n                        "text": "A sample piece of text 2"\n                    }\n                ]\n            }\n        },\n        "parameters": {\n            "to_case": "upper"\n        }\n    }\'\n    \'http://localhost:5000/transform_case\'\n'})}),"\n",(0,a.jsx)(n.p,{children:"which produces the following response:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-json",children:'{\n    "output_type": "batchtext",\n    "texts": [\n        {\n            "output_type": "text",\n            "value": "A SAMPLE PIECE OF TEXT 1",\n            "title": "a sample piece of text 1",\n            "subtitle": null\n        },\n        {\n            "output_type": "text",\n            "value": "A SAMPLE PIECE OF TEXT 2",\n            "title": "a sample piece of text 2",\n            "subtitle": null\n        }\n    ]\n}\n'})}),"\n",(0,a.jsx)(n.p,{children:"Great! We got our first task running!"}),"\n",(0,a.jsx)(n.h3,{id:"adding-a-ui-schema",children:"Adding a UI Schema"}),"\n",(0,a.jsxs)(n.p,{children:["To define a UI for this sample model, we need to write a task schema function. This function will return a ",(0,a.jsx)(n.code,{children:"TaskSchema"})," object, which contains specifications for the types and possible values for all the inputs and parameters for our ML function."]}),"\n",(0,a.jsxs)(n.p,{children:["First, we will write a schema for our inputs by building an ",(0,a.jsx)(n.code,{children:"InputSchema"}),". We simply specify a ",(0,a.jsx)(n.code,{children:"key"})," (which must match the key for the respective attribute in the ",(0,a.jsx)(n.code,{children:"TypedDict"})," defined for our inputs), a ",(0,a.jsx)(n.code,{children:"label"})," to be displayed on the UI, and the ",(0,a.jsx)(n.code,{children:"input_type"}),", which must be one of the following:"]}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.TEXT"}),": a single raw text input"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.FILE"}),": a single file path"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.DIRECTORY"}),": a single directory path"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.MARKDOWN"}),": a markdown string"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.BATCHTEXT"}),": a collection of raw text inputs"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.BATCHFILE"}),": a collection of file paths"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"InputType.BATCHDIRECTORY"}),": a collection of directory paths"]}),"\n"]}),"\n",(0,a.jsx)(n.p,{children:"Let's write an input schema for our function's inputs:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'inputSchema = InputSchema(\n    key="text_inputs",\n    label="Text to Transform",\n    input_type=InputType.BATCHTEXT\n)\n'})}),"\n",(0,a.jsxs)(n.p,{children:["Next, we will write a similar schema for our parameters by building an ",(0,a.jsx)(n.code,{children:"ParameterSchema"}),". We again specify a ",(0,a.jsx)(n.code,{children:"key"})," (which must match the key for the respective attribute in the ",(0,a.jsx)(n.code,{children:"TypedDict"})," defined for our parameters), a ",(0,a.jsx)(n.code,{children:"label"})," to be displayed on the UI, an optional ",(0,a.jsx)(n.code,{children:"subtitle"})," that is more expansive than the label, and the ",(0,a.jsx)(n.code,{children:"value"}),", which must be an instance of one of these types:"]}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"TextParameterDescriptor"}),": a string of raw text","\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"default"}),": default value"]}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"EnumParameterDescriptor"}),": an enumeration with multiple values","\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"enum_vals"}),": list of ",(0,a.jsx)(n.code,{children:"EnumVal"})," objects, each containing a unique ",(0,a.jsx)(n.code,{children:"key"})," and ",(0,a.jsx)(n.code,{children:"label"})]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"message_when_empty"}),": message to display on UI if this enum contains no values (optional)"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"default"}),": default value"]}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"IntParameterDescriptor"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"default"}),": default value"]}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"RangedIntParameterDescriptor"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"default"}),": default value"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"range"}),": an ",(0,a.jsx)(n.code,{children:"IntRangeDescriptor"})," containing a ",(0,a.jsx)(n.code,{children:"min"})," value and ",(0,a.jsx)(n.code,{children:"max"})," value"]}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"FloatParameterDescriptor"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"default"}),": default value"]}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"RangedFloatParameterDescriptor"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"default"}),": default value"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"range"}),": a ",(0,a.jsx)(n.code,{children:"FloatRangeDescriptor"})," containing a ",(0,a.jsx)(n.code,{children:"min"})," value and ",(0,a.jsx)(n.code,{children:"max"})," value"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,a.jsx)(n.p,{children:"Let's write a parameter schema for our function's parameters:"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'parameterSchema = ParameterSchema(\n    key="to_case",\n    label="Case to Transform Text Into",\n    subtitle="\'upper\' will convert all text to upper case. \'lower\' will convert all text to lower case.",\n    value=EnumParameterDescriptor(\n        enum_vals=[\n            EnumVal(\n                key="upper",\n                label="UPPER"\n            ),\n            EnumVal(\n                key="lower",\n                label="LOWER"\n            )\n        ],\n        default="upper"\n    )\n)\n'})}),"\n",(0,a.jsxs)(n.p,{children:["Let's use the above schemas to define a UI schema for our ",(0,a.jsx)(n.code,{children:"transform_case"})," task:"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"from flask_ml.flask_ml_server.models import TaskSchema, InputSchema, ParameterSchema, InputType, EnumParameterDescriptor, EnumVal\n\ndef create_transform_case_task_schema() -> TaskSchema:\n    return TaskSchema(\n        inputs = [inputSchema],\n        parameters = [parameterSchema]\n    )\n"})}),"\n",(0,a.jsxs)(n.p,{children:["Now, we can register this function in our endpoint by setting the ",(0,a.jsx)(n.code,{children:"task_schema_func"})," parameter in the route annotation:"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'@server.route("/transform_case", task_schema_func=create_transform_case_task_schema)\ndef transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:\n    ...\n'})}),"\n",(0,a.jsx)(n.p,{children:"We can also set a few other parameters within the annotation in order to provide the UI some extra information:"}),"\n",(0,a.jsxs)(n.ul,{children:["\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"short_title"}),": concise name for your task, to be displayed as a heading in the UI"]}),"\n",(0,a.jsxs)(n.li,{children:[(0,a.jsx)(n.code,{children:"order"}),": if you support multiple ML tasks (endpoints), this defines the order in which the task tabs should appear in the UI (starts from 0)"]}),"\n"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'@server.route("/transform_case", task_schema_func=create_transform_case_task_schema, short_title="Transform Case", order=0)\ndef transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:\n    ...\n'})}),"\n",(0,a.jsxs)(n.p,{children:["Note that if you call ",(0,a.jsx)(n.code,{children:"/api/routes"})," again, you will get a more extensive response this time:"]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-json",children:'[\n    {\n        "order": 0,\n        "payload_schema": "/transform_case/payload_schema",\n        "run_task": "/transform_case",\n        "sample_payload": "/transform_case/sample_payload",\n        "short_title": "Transform Case",\n        "task_schema": "/transform_case/task_schema"\n    }\n]\n'})}),"\n",(0,a.jsxs)(n.p,{children:["Note that calling the ",(0,a.jsx)(n.code,{children:"/transform_case/task_schema"})," route simply executes the TaskSchema function that we had written and returns its output as a response. This contains all the information that the UI will need in order to determine how to render each of the inputs and parameters in its forms."]}),"\n",(0,a.jsx)(n.h2,{id:"entire-sample-code",children:"Entire Sample Code"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'from typing import TypedDict\nfrom flask_ml.flask_ml_server import MLServer\nfrom flask_ml.flask_ml_server.models import BatchTextInput, BatchTextResponse, EnumParameterDescriptor, EnumVal, InputSchema, InputType, ParameterSchema, ResponseBody, TaskSchema, TextResponse\n\nserver = MLServer(__name__)\n\nclass TransformCaseInputs(TypedDict):\n    text_inputs: BatchTextInput\n\nclass TransformCaseParameters(TypedDict):\n    to_case: str # \'upper\' or \'lower\'\n\ndef create_transform_case_task_schema() -> TaskSchema:\n    input_schema = InputSchema(\n        key="text_inputs",\n        label="Text to Transform",\n        input_type=InputType.BATCHTEXT\n    )\n    parameter_schema = ParameterSchema(\n        key="to_case",\n        label="Case to Transform Text Into",\n        subtitle="\'upper\' will convert all text to upper case. \'lower\' will convert all text to lower case.",\n        value=EnumParameterDescriptor(\n            enum_vals=[\n                EnumVal(\n                    key="upper",\n                    label="UPPER"\n                ),\n                EnumVal(\n                    key="lower",\n                    label="LOWER"\n                )\n            ],\n            default="upper"\n        )\n    )\n    return TaskSchema(\n        inputs = [input_schema],\n        parameters = [parameter_schema]\n    )\n\n@server.route(\n    "/transform_case",\n    task_schema_func=create_transform_case_task_schema,\n    short_title="Transform Case",\n    order=0\n)\ndef transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:\n    to_upper: bool = parameters[\'to_case\'] == \'upper\'\n    \n    outputs = []\n    for text_input in inputs[\'text_inputs\'].texts:\n        raw_text = text_input.text\n        processed_text = raw_text.upper() if to_upper else raw_text.lower()\n        outputs.append(TextResponse(value=processed_text, title=raw_text))\n\n    return ResponseBody(root=BatchTextResponse(texts=outputs))\n\nif __name__ == "__main__":\n    # Run a debug server\n    server.run()\n\n'})}),"\n",(0,a.jsx)(n.h1,{id:"additional-features",children:"Additional Features"}),"\n",(0,a.jsx)(n.h2,{id:"adding-a-automatically-generated-cli",children:"Adding a automatically generated CLI"}),"\n",(0,a.jsxs)(n.p,{children:["Flask-ML can automatically generate a CLI for your machine learning code. See ",(0,a.jsx)(n.a,{href:"./cli",children:"Writing a CLI"})," for more information."]}),"\n",(0,a.jsx)(n.h2,{id:"adding-application-metadata",children:"Adding Application Metadata"}),"\n",(0,a.jsxs)(n.p,{children:["You can provide metadata about your application for use by ",(0,a.jsx)(n.a,{href:"https://github.com/UMass-Rescue/RescueBox-Desktop",children:"Rescue-Box Desktop"}),". See ",(0,a.jsx)(n.a,{href:"./metadata",children:"Adding Application Metadata"})," for more information."]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,a.jsx)(n,{...e,children:(0,a.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>i,x:()=>o});var s=t(6540);const a={},r=s.createContext(a);function i(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:i(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);