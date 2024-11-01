from pathlib import Path
from re import T
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask.testing import FlaskClient
import pytest

TEST_MARKDOWN_FILE_PATH = "test_markdown_file_path.md"
TEST_MARKDOWN_FILE_CONTENT = "# This is a test markdown file"


@pytest.fixture
def test_markdown_file_path(tmp_path: Path) -> str:
    file_path = tmp_path / TEST_MARKDOWN_FILE_PATH
    file_path.write_text(TEST_MARKDOWN_FILE_CONTENT)
    return str(file_path)


def test_app_metadata_provided(server: MLServer, app: FlaskClient, test_markdown_file_path: str):
    server.add_app_metadata(
        info=load_file_as_string(test_markdown_file_path),
        author="Test Author",
        version="1.0.0",
        name="Test App",
    )

    response = app.get("/api/app_metadata")
    assert response.status_code == 200
    assert response.json == {
        "info": TEST_MARKDOWN_FILE_CONTENT,
        "author": "Test Author",
        "version": "1.0.0",
        "name": "Test App",
    }


def test_app_metadata_not_provided(app: FlaskClient):
    response = app.get("/api/app_metadata")
    assert response.status_code == 200
    assert response.json == {"error": "App metadata not set"}


def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        load_file_as_string("invalid_file_path.md")
