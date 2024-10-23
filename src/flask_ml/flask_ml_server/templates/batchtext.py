from typing import Callable, List, Optional
from typing_extensions import TypedDict
from ...flask_ml_server import models


class TextInputs(TypedDict):
    text_inputs: models.BatchTextInput


def create_task_schema_func(
    parameter_schemas: Optional[List[models.ParameterSchema]] = None,
) -> Callable[[], models.TaskSchema]:
    if parameter_schemas is None:
        parameter_schemas = []

    return lambda: models.TaskSchema(
        inputs=[
            models.InputSchema(
                key="text_inputs", label="Provide text inputs", input_type=models.InputType.BATCHTEXT
            ),
        ],
        parameters=parameter_schemas,
    )
