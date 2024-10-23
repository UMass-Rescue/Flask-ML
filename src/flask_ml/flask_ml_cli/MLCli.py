from argparse import _SubParsersAction, ArgumentParser
from email.policy import default
import json
from typing import Callable, Text
from typing_extensions import assert_never

from flask_ml.flask_ml_cli.utils import (
    is_path_exists_or_creatable_portable_arg_parser,
)
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.MLServer import EndpointDetails
from flask_ml.flask_ml_server.models import (
    BatchDirectoryInput,
    BatchFileInput,
    BatchTextInput,
    DirectoryInput,
    FileInput,
    InputSchema,
    InputType,
    ParameterSchema,
    ParameterType,
    ResponseBody,
    TaskSchema,
    TextInput,
    EnumParameterDescriptor,
)


def get_input_argument_validator_func(input_type: InputType):
    match input_type:
        case InputType.FILE:
            return is_path_exists_or_creatable_portable_arg_parser
        case InputType.DIRECTORY:
            return is_path_exists_or_creatable_portable_arg_parser
        case InputType.TEXT:
            return str
        case InputType.TEXTAREA:
            return str
        case InputType.BATCHFILE:
            return is_path_exists_or_creatable_portable_arg_parser
        case InputType.BATCHTEXT:
            return str
        case InputType.BATCHDIRECTORY:
            return is_path_exists_or_creatable_portable_arg_parser
        case _:
            assert_never(input_type)


def get_parameter_argument_validator_func(parameter_type: ParameterType):
    match parameter_type:
        case ParameterType.RANGED_FLOAT:
            return float
        case ParameterType.FLOAT:
            return float
        case ParameterType.ENUM:
            return str
        case ParameterType.TEXT:
            return str
        case ParameterType.RANGED_INT:
            return int
        case ParameterType.INT:
            return int
        case _:
            assert_never(parameter_type)


def get_enum_parameter_choices(parameter_schema: ParameterSchema):
    assert isinstance(parameter_schema.value, EnumParameterDescriptor)
    return [item.key for item in parameter_schema.value.enum_vals]


class MLCli:
    def __init__(self, server: MLServer, argument_parser: ArgumentParser, verbose=False):
        self._server = server
        self._parser = argument_parser
        self._verbose = verbose

    def _get_name_of_subcommand(self, endpoint: EndpointDetails):
        rule = endpoint.rule
        if rule.startswith("/"):
            rule = rule[1:]
        return rule.replace("/", "_")

    def _add_input_argument_to_parser(self, parser: ArgumentParser, input_schema: InputSchema):
        name = "--" + input_schema.key
        help = input_schema.subtitle if input_schema.subtitle else input_schema.label
        input_type = input_schema.input_type

        # Figure out if inputs could be one or more values
        nargs = None
        if input_type in [InputType.BATCHFILE, InputType.BATCHDIRECTORY, InputType.BATCHTEXT]:
            nargs = "+"

        parser.add_argument(
            name, help=help, required=True, type=get_input_argument_validator_func(input_type), nargs=nargs
        )

    def _add_parameter_argument_to_parser(self, parser: ArgumentParser, parameter_schema: ParameterSchema):
        name = "--" + parameter_schema.key
        help = parameter_schema.subtitle if parameter_schema.subtitle else parameter_schema.label
        parameter_type = parameter_schema.value.parameter_type
        if parameter_type is None:
            raise ValueError("FATAL: Parameter type must never be None")

        default_param_value = parameter_schema.value.default
        if default_param_value is not None:
            parser.add_argument(
                name,
                help=help,
                default=default_param_value,
                type=get_parameter_argument_validator_func(parameter_type),
                choices=(
                    get_enum_parameter_choices(parameter_schema)
                    if parameter_type == ParameterType.ENUM
                    else None
                ),
            )
        else:
            parser.add_argument(
                name, help=help, required=True, type=get_parameter_argument_validator_func(parameter_type)
            )

    def _set_function_on_parser(self, parser, task_schema: TaskSchema, ml_func: Callable[..., ResponseBody]):
        def func(args):
            inputs = {}
            parameters = {}
            for input_schema in task_schema.inputs:
                cli_input = getattr(args, input_schema.key)
                match input_schema.input_type:
                    case InputType.FILE:
                        input_model = FileInput(path=cli_input)
                    case InputType.DIRECTORY:
                        input_model = DirectoryInput(path=cli_input)
                    case InputType.TEXT:
                        input_model = TextInput(text=cli_input)
                    case InputType.TEXTAREA:
                        input_model = TextInput(text=cli_input)
                    case InputType.BATCHFILE:
                        input_model = BatchFileInput(files=[FileInput(path=item) for item in cli_input])
                    case InputType.BATCHTEXT:
                        input_model = BatchTextInput(texts=[TextInput(text=item) for item in cli_input])
                    case InputType.BATCHDIRECTORY:
                        input_model = BatchDirectoryInput(
                            directories=[DirectoryInput(path=item) for item in cli_input]
                        )
                    case _:
                        assert_never(input_schema.input_type)
                inputs[input_schema.key] = input_model
            for parameter_schema in task_schema.parameters:
                parameters[parameter_schema.key] = getattr(args, parameter_schema.key)
            result = ml_func(inputs, parameters)
            print(json.dumps(json.loads(result.model_dump_json()), indent=4))

        parser.set_defaults(func=func)

    def _add_subparser(self, subparsers: _SubParsersAction, endpoint: EndpointDetails):
        name = self._get_name_of_subcommand(endpoint)
        help = endpoint.short_title

        task_schema = endpoint.task_schema_func()

        input_schemas = task_schema.inputs
        parameter_schemas = task_schema.parameters

        subcommand_parser = subparsers.add_parser(name, help=help)

        for input_schema in input_schemas:
            self._add_input_argument_to_parser(subcommand_parser, input_schema)
        for parameter_schema in parameter_schemas:
            self._add_parameter_argument_to_parser(subcommand_parser, parameter_schema)
        self._set_function_on_parser(subcommand_parser, task_schema, endpoint.func)

    def run_cli(self):
        schema_endpoints = [
            endpoint for endpoint in self._server.endpoints if isinstance(endpoint, EndpointDetails)
        ]

        if len(schema_endpoints) == 0:
            if self._verbose:
                print("FATAL: At least one schema endpoint must be defined")
            raise ValueError("This model does not support the CLI. Run with verbose=True to see the error")

        subparsers = self._parser.add_subparsers(help="Subcommands", required=True)
        for endpoint in schema_endpoints:
            self._add_subparser(subparsers, endpoint)

        args = self._parser.parse_args()
        if args.func:
            print()
            args.func(args)
