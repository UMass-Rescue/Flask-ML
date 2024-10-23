from typing import Callable, List, Optional
from typing_extensions import TypedDict
from ...flask_ml_server import models


class FileInputs(TypedDict):
    file_inputs: models.BatchFileInput


def create_task_schema_func(
    parameter_schemas: Optional[List[models.ParameterSchema]] = None,
) -> Callable[[], models.TaskSchema]:
    if parameter_schemas is None:
        parameter_schemas = []

    return lambda: models.TaskSchema(
        inputs=[
            models.InputSchema(
                key="file_inputs", label="Provide file inputs", input_type=models.InputType.BATCHFILE
            ),
        ],
        parameters=parameter_schemas,
    )
