from pathlib import Path
from typing import Optional



class AudioTranscriptionModel:
    def __init__(self, model_path: str = "base"):
        import whisper
        self.model = whisper.load_model(model_path)
        self.audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}

    def get_audio_files(self, directory: str) -> list[Path]:
        audio_files: list[Path] = []

        # Convert string path to Path object
        directory_path = Path(directory)

        # Iterate over files in directory and subdirectories
        for file_path in directory_path.rglob("*"):
            if file_path.suffix.lower() in self.audio_extensions:
                audio_files.append(file_path)

        return audio_files

    def _validate_audio_path(self, audio_path: str) -> None:
        if audio_path is None:
            raise ValueError("audio_path cannot be None")

    def transcribe(self, audio_path: str, out_dir: Optional[str] = None) -> str:
        self._validate_audio_path(audio_path)
        res: str = self.model.transcribe(str(audio_path))["text"]  # type: ignore
        if out_dir:
            self._write_res_to_dir([{"file_path": str(audio_path), "result": res}], out_dir)
        return res

    def transcribe_batch(self, audio_paths: list[str]) -> list[dict[str, str]]:
        return [
            {"file_path": str(audio_path), "result": self.transcribe(str(audio_path))}
            for audio_path in audio_paths
        ]

    def _write_res_to_dir(self, res: list[dict[str, str]], out_dir: str) -> None:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        for r in res:
            with open(out_dir_path / (r["file_path"].split("/")[-1].split(".")[0] + ".txt"), "w") as f:
                f.write(r["result"])

    def transcribe_files_in_directory(
        self, input_dir: str, out_dir: Optional[str] = None
    ) -> list[dict[str, str]]:
        res = self.transcribe_batch([str(file) for file in self.get_audio_files(input_dir)])
        if out_dir:
            self._write_res_to_dir(res, out_dir)
        return res
