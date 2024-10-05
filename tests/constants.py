FILE_INPUT_SCHEMA = {
    "items": {
        "properties": {
        "file_path": {
            "description": "Path of the file to be processed",
            "minLength": 1,
            "title": "File Path",
            "type": "string"
        }
        },
        "required": [
        "file_path"
        ],
        "title": "FileInput",
        "type": "object"
    },
    "type": "array"
}


TEXT_INPUT_SCHEMA = {
    "items": {
        "properties": {
        "text": {
            "description": "Text to be processed",
            "minLength": 1,
            "title": "Text",
            "type": "string"
        }
        },
        "required": [
        "text"
        ],
        "title": "TextInput",
        "type": "object"
    },
    "type": "array"
}

RESPONSE_MODEL_OUTPUT_SCHEMA = {
    "$defs": {
        "AudioResult": {
        "properties": {
            "file_path": {
            "description": "Path of the file associated with the result",
            "minLength": 1,
            "title": "File Path",
            "type": "string"
            },
            "result": {
            "description": "The result, which can be any JSON-serializable object",
            "minLength": 1,
            "title": "Result"
            }
        },
        "required": [
            "result",
            "file_path"
        ],
        "title": "AudioResult",
        "type": "object"
        },
        "ImageResult": {
        "properties": {
            "file_path": {
            "description": "Path of the file associated with the result",
            "minLength": 1,
            "title": "File Path",
            "type": "string"
            },
            "result": {
            "description": "The result, which can be any JSON-serializable object",
            "minLength": 1,
            "title": "Result"
            }
        },
        "required": [
            "result",
            "file_path"
        ],
        "title": "ImageResult",
        "type": "object"
        },
        "TextResult": {
        "properties": {
            "result": {
            "description": "The result, which can be any JSON-serializable object",
            "minLength": 1,
            "title": "Result"
            },
            "text": {
            "description": "The text content associated with the result",
            "minLength": 1,
            "title": "Text",
            "type": "string"
            }
        },
        "required": [
            "result",
            "text"
        ],
        "title": "TextResult",
        "type": "object"
        },
        "VideoResult": {
        "properties": {
            "file_path": {
            "description": "Path of the file associated with the result",
            "minLength": 1,
            "title": "File Path",
            "type": "string"
            },
            "result": {
            "description": "The result, which can be any JSON-serializable object",
            "minLength": 1,
            "title": "Result"
            }
        },
        "required": [
            "result",
            "file_path"
        ],
        "title": "VideoResult",
        "type": "object"
        }
    },
    "description": "Model representing the results from an ML model.\nAttributes:\n    status (str): The status of the operation, e.g., 'success'\n    results (List[MLResult]): List of results.\nMethods:\n    get_response(status_code: int = 200) -\u003E Response:\n        Returns a Flask Response object with the JSON representation of the model.",
    "properties": {
        "results": {
        "description": "List of results, each either a file or text with its result",
        "items": {
            "anyOf": [
            {
                "$ref": "#/$defs/TextResult"
            },
            {
                "$ref": "#/$defs/ImageResult"
            },
            {
                "$ref": "#/$defs/AudioResult"
            },
            {
                "$ref": "#/$defs/VideoResult"
            }
            ]
        },
        "minItems": 1,
        "title": "Results",
        "type": "array"
        },
        "status": {
        "default": "SUCCESS",
        "description": "The status of the operation, e.g., 'SUCCESS'",
        "minLength": 1,
        "title": "Status",
        "type": "string"
        }
    },
    "required": [
        "results"
    ],
    "title": "ResponseModel",
    "type": "object"
}