from typing import Union
from flask_ml.flask_ml_server.models import (
    FileResponse,
    DirectoryResponse,
    MarkdownResponse,
    ResponseBody,
    TextResponse,
    BatchFileResponse,
    BatchTextResponse,
    BatchDirectoryResponse,
)


def response_body(
    result: Union[
        FileResponse,
        DirectoryResponse,
        MarkdownResponse,
        TextResponse,
        BatchFileResponse,
        BatchTextResponse,
        BatchDirectoryResponse,
    ]
):
    return ResponseBody(root=result)
