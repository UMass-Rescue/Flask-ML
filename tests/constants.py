FILE_INPUT_SCHEMA = {
    "items": {
        "properties": {
            "file_path": {
                "description": "Path of the file to be processed",
                "minLength": 1,
                "title": "File Path",
                "type": "string",
            }
        },
        "required": ["file_path"],
        "title": "FileInput",
        "type": "object",
    },
    "type": "array",
}


TEXT_INPUT_SCHEMA = {
    "items": {
        "properties": {
            "text": {
                "description": "Text to be processed",
                "minLength": 1,
                "title": "Text",
                "type": "string",
            }
        },
        "required": ["text"],
        "title": "TextInput",
        "type": "object",
    },
    "type": "array",
}

BATCH_TEXT_RESPONSE_SCHEMA = {
    "$defs": {
        "TextResult": {
            "properties": {
                "id": {
                    "description": "The ID of the result",
                    "minLength": 1,
                    "title": "Id",
                    "type": "string",
                },
                "result": {
                    "description": "The result text.",
                    "minLength": 1,
                    "title": "Result",
                    "type": "string",
                },
            },
            "required": ["id", "result"],
            "title": "TextResult",
            "type": "object",
        }
    },
    "properties": {
        "results": {
            "description": "List of text results",
            "items": {"$ref": "#/$defs/TextResult"},
            "minItems": 1,
            "title": "Results",
            "type": "array",
        }
    },
    "required": ["results"],
    "title": "BatchTextResult",
    "type": "object",
}


BATCH_IMAGE_RESPONSE_SCHEMA = {
    "$defs": {
        "ImageResult": {
            "properties": {
                "id": {
                    "description": "The ID of the result",
                    "minLength": 1,
                    "title": "Id",
                    "type": "string",
                },
                "result": {
                    "description": "Path of the result file.",
                    "minLength": 1,
                    "title": "Result",
                    "type": "string",
                },
            },
            "required": ["id", "result"],
            "title": "ImageResult",
            "type": "object",
        }
    },
    "properties": {
        "results": {
            "description": "List of image results",
            "items": {"$ref": "#/$defs/ImageResult"},
            "minItems": 1,
            "title": "Results",
            "type": "array",
        }
    },
    "required": ["results"],
    "title": "BatchImageResult",
    "type": "object",
}
