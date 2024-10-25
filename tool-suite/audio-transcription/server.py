from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *

from .model import AudioTranscriptionModel

server = MLServer(__name__)


class TranscriptionInputs(TypedDict):
    audio_files: BatchFileInput


class NoParameters(TypedDict):
    pass


def task_schema_func():
    return TaskSchema(
        inputs=[
            InputSchema(
                key="audio_files",
                label="Audio Files",
                subtitle="Select the audio files to transcribe",
                input_type=InputType.BATCHFILE,
            )
        ],
        parameters=[],
    )


@server.route("/transcribe", task_schema_func=task_schema_func)
def transcribe(inputs: TranscriptionInputs, parameters: NoParameters) -> ResponseBody:
    print("Inputs:", inputs)
    print()
    print("Parameters:", parameters)
    print()
    files = [e.path for e in inputs["audio_files"].files]
    
    model = AudioTranscriptionModel()
    results = model.transcribe_batch(files)
    
    results = {r["file_path"]: r["result"] for r in results}
    return ResponseBody(
        root=(
            BatchTextResponse(
                texts=[
                    TextResponse(
                        title=k,
                        value=v,
                    )
                    for k, v in results.items()
                ]
            )
        )
    )


if __name__ == "__main__":
    server.run()
