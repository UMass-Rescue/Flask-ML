import argparse
from flask_ml.flask_ml_cli import MLCli
from flask_ml.flask_ml_server import MLServer
import pytest


@pytest.fixture
def ml_cli(server: MLServer):
    argparser = argparse.ArgumentParser()
    ml_cli = MLCli(server, argparser)
    ml_cli._setup_cli()
    return ml_cli

def test_run_cli_fail_no_subcommand(server: MLServer):
    argparser = argparse.ArgumentParser()
    ml_cli = MLCli(server, argparser)
    with pytest.raises(SystemExit):
        ml_cli.run_cli()

def test_run_cli(server: MLServer):
    argparser = argparse.ArgumentParser()
    ml_cli = MLCli(server, argparser)
    ml_cli.run_cli(["process_texts_with_schema", "--text_inputs", "text1.txt", "text2.txt", "--param1", "4"])

def test_ml_cli_with_no_valid_endpoint():
    server = MLServer(__name__)
    argparser = argparse.ArgumentParser()
    ml_cli = MLCli(server, argparser, verbose=True)
    with pytest.raises(ValueError) as e:
        ml_cli.run_cli()
    assert e.exconly() == "ValueError: This model does not support the CLI. Run with verbose=True to see the error"

def test_arg_parser_has_all_subcommands(ml_cli: MLCli):
    assert set(ml_cli._parser._actions[1].choices.keys()) == {  # type: ignore
        "process_invalid_with_schema",
        "process_texts_with_schema",
        "process_text_with_schema",
        "process_files_with_schema",
        "process_file_with_schema",
        "process_directory_and_enum_parameter_with_schema",
        "process_directories_and_ranged_int_parameter_with_schema",
        "process_text_input_with_text_area_schema"
    }


def test_process_file_with_schema_invalid_path(ml_cli: MLCli):
    with pytest.raises(Exception):
        parsed_args = ml_cli._parse_args(
            ["process_invalid_with_schema", "--file_input", "invalid_path.txt", "--param1", "0.5"]
        )
        ml_cli._run_cli_and_return(parsed_args)

# Single Text Input

def test_process_text_with_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        ["process_text_with_schema", "--text_input", "text.txt", "--param1", "ABCD"]
    )

    assert parsed_args.text_input == "text.txt"
    assert parsed_args.param1 == "ABCD"
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode="json") == {
        "output_type": "text",
        "value": "processed_text.txt",
        "title": "text.txt",
        "subtitle": None,
    }

def test_process_text_with_schema_no_input_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_text_with_schema"]
        )
        ml_cli._run_cli_and_return(parsed_args)

# Texts

def test_process_texts_with_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        ["process_texts_with_schema", "--text_inputs", "text1.txt", "text2.txt", "--param1", "4"]
    )

    assert parsed_args.text_inputs == ["text1.txt", "text2.txt"]
    assert parsed_args.param1 == 4
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode="json") == {
        "output_type": "batchtext",
        "texts": [
            {"output_type": "text", "value": "processed_text.txt", "title": "text1.txt", "subtitle": None},
            {"output_type": "text", "value": "processed_text.txt", "title": "text2.txt", "subtitle": None},
        ],
    }

def test_process_texts_with_schema_no_input_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_texts_with_schema", "--text_inputs"]
        )
        ml_cli._run_cli_and_return(parsed_args)

def test_process_texts_with_schema_non_int_parameter_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_texts_with_schema", "--text_inputs", "text1.txt", "text2.txt", "--param1", "ABCD"]
        )
        ml_cli._run_cli_and_return(parsed_args)

# Text Input with Text Area

def test_process_text_input_with_text_area_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        ["process_text_input_with_text_area_schema", "--text_input", "text.txt", "--param1", "ABCD"]
    )

    assert parsed_args.text_input == "text.txt"
    assert parsed_args.param1 == "ABCD"
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode="json") == {
        "output_type": "text",
        "value": "processed_text.txt",
        "title": "text.txt",
        "subtitle": None,
    }

# Files

def test_process_files_with_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        ["process_files_with_schema", "--file_inputs", "file1.txt", "file2.txt", "--param1", "0.5"]
    )

    assert parsed_args.file_inputs == ["file1.txt", "file2.txt"]
    assert parsed_args.param1 == 0.5
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode="json") == {
        "output_type": "batchfile",
        "files": [
            {
                "output_type": "file",
                "file_type": "img",
                "path": "processed_image.img",
                "title": "file1.txt",
                "subtitle": None,
            },
            {
                "output_type": "file",
                "file_type": "img",
                "path": "processed_image.img",
                "title": "file2.txt",
                "subtitle": None,
            },
        ],
    }

def test_process_files_with_schema_no_input_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_files_with_schema", "--file_inputs"]
        )
        ml_cli._run_cli_and_return(parsed_args)

def test_process_files_with_schema_out_of_range_float_parameter_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_files_with_schema", "--file_inputs", "file1.txt", "file2.txt", "--param1", "2.0"]
        )
        ml_cli._run_cli_and_return(parsed_args)

def test_process_files_with_schema_invalid_float_parameter_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_files_with_schema", "--file_inputs", "file1.txt", "file2.txt", "--param1", "NOT_A_FLOAT"]
        )
        ml_cli._run_cli_and_return(parsed_args)

# Single File

def test_process_file_with_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        ["process_file_with_schema", "--file_input", "file_path.txt", "--param1", "0.5"]
    )
    assert parsed_args.file_input == "file_path.txt"
    assert parsed_args.param1 == 0.5
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode="json") == {
        "output_type": "file",
        "file_type": "img",
        "path": "processed_image.img",
        "title": "file_path.txt",
        "subtitle": None,
    }

def test_process_file_with_schema_no_input_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_file_with_schema"]
        )
        ml_cli._run_cli_and_return(parsed_args)

# Directory

def test_process_directory_and_enum_parameter_with_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        [
            "process_directory_and_enum_parameter_with_schema",
            "--dir_input",
            "./dir_path",
            "--param1",
            "option_1",
        ]
    )
    assert parsed_args.dir_input == "./dir_path"
    assert parsed_args.param1 == "option_1"
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode="json") == {
        "output_type": "directory",
        "path": "processed_directory",
        "title": "./dir_path",
        "subtitle": None,
    }

def test_process_directory_and_enum_parameter_with_schema_no_input_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_directory_and_enum_parameter_with_schema", "--dir_input"]
        )
        ml_cli._run_cli_and_return(parsed_args)

def test_process_directory_and_enum_parameter_with_schema_non_enum_parameter_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            ["process_directory_and_enum_parameter_with_schema", "--dir_input", "./dir_path", "--param1", "INVALID_OPTION"]
        )
        ml_cli._run_cli_and_return(parsed_args)

# Directories

def test_process_directories_and_ranged_int_parameter_with_schema(ml_cli: MLCli):
    parsed_args = ml_cli._parse_args(
        [
            "process_directories_and_ranged_int_parameter_with_schema",
            "--dir_inputs",
            "./dir_path1",
            "./dir_path2",
            "--param1",
            "1",
        ]
    )
    assert parsed_args.dir_inputs == ["./dir_path1", "./dir_path2"]
    assert parsed_args.param1 == 1
    assert parsed_args.func is not None

    response = ml_cli._run_cli_and_return(parsed_args)
    assert response is not None
    assert response.model_dump(mode = 'json') == {
        "output_type": "batchdirectory",
        "directories": [
            {
                "output_type": "directory",
                "path": "processed_directory",
                "title": "./dir_path1",
                "subtitle": None,
            },
            {
                "output_type": "directory",
                "path": "processed_directory",
                "title": "./dir_path2",
                "subtitle": None,
            },
        ],
    }

def test_process_directories_and_ranged_int_parameter_with_schema_out_of_range_int_parameter_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            [
                "process_directories_and_ranged_int_parameter_with_schema",
                "--dir_inputs",
                "./dir_path1",
                "./dir_path2",
                "--param1",
                "1000000",
            ]
        )
        ml_cli._run_cli_and_return(parsed_args)

def test_process_directories_and_ranged_int_parameter_with_schema_non_int_parameter_provided(ml_cli: MLCli):
    with pytest.raises(SystemExit):
        parsed_args = ml_cli._parse_args(
            [
                "process_directories_and_ranged_int_parameter_with_schema",
                "--dir_inputs",
                "./dir_path1",
                "./dir_path2",
                "--param1",
                "NOT_AN_INT",
            ]
        )
        ml_cli._run_cli_and_return(parsed_args)
