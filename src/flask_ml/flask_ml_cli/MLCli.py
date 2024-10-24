from argparse import ArgumentParser, Namespace
import json
from typing import Callable, Optional, Sequence
from typing_extensions import assert_never

from flask_ml.flask_ml_cli.utils import (
    get_float_range_check_func_arg_parser,
    get_int_range_check_func_arg_parser,
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
    FloatParameterDescriptor,
    InputSchema,
    InputType,
    IntParameterDescriptor,
    ParameterSchema,
    ParameterType,
    RangedFloatParameterDescriptor,
    RangedIntParameterDescriptor,
    ResponseBody,
    TaskSchema,
    TextInput,
    EnumParameterDescriptor,
    TextParameterDescriptor,
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
        case _: # pragma: no cover
            assert_never(input_type)


def get_parameter_argument_validator_func(parameter_schema: ParameterSchema):
    match parameter_schema.value:
        case RangedFloatParameterDescriptor():
            return get_float_range_check_func_arg_parser(parameter_schema.value.range)
        case FloatParameterDescriptor():
            return float
        case EnumParameterDescriptor():
            return str
        case TextParameterDescriptor():
            return str
        case RangedIntParameterDescriptor():
            return get_int_range_check_func_arg_parser(parameter_schema.value.range)
        case IntParameterDescriptor():
            return int
        case _: # pragma: no cover
            assert_never(parameter_schema.value)


def get_enum_parameter_choices(parameter_schema: ParameterSchema):
    assert isinstance(parameter_schema.value, EnumParameterDescriptor)
    return [item.key for item in parameter_schema.value.enum_vals]


class MLCli:
    def __init__(self, server: MLServer, argument_parser: ArgumentParser, verbose=False):
        self._server = server
        self._parser = argument_parser
        self._verbose = verbose

    @staticmethod
    def _get_name_of_subcommand(endpoint: EndpointDetails):
        rule = endpoint.rule
        if rule.startswith("/"):
            rule = rule[1:]
        return rule.replace("/", "_")

    @staticmethod
    def _add_input_argument_to_parser(parser: ArgumentParser, input_schema: InputSchema):
        name = "--" + input_schema.key
        helpp = input_schema.subtitle if input_schema.subtitle else input_schema.label
        input_type = input_schema.input_type

        # Figure out if inputs could be one or more values
        nargs = None
        if input_type in [InputType.BATCHFILE, InputType.BATCHDIRECTORY, InputType.BATCHTEXT]:
            nargs = "+"

        parser.add_argument(
            name, help=helpp, required=True, type=get_input_argument_validator_func(input_type), nargs=nargs
        )

    @staticmethod
    def _add_parameter_argument_to_parser(parser: ArgumentParser, parameter_schema: ParameterSchema):
        name = "--" + parameter_schema.key
        helpp = parameter_schema.subtitle if parameter_schema.subtitle else parameter_schema.label
        parameter_type = parameter_schema.value.parameter_type
        if parameter_type is None:
            raise ValueError("FATAL: Parameter type must never be None")  # pragma: no cover

        default_param_value = parameter_schema.value.default
        parser.add_argument(
            name,
            help=helpp,
            default=default_param_value,
            type=get_parameter_argument_validator_func(parameter_schema),
            choices=(
                get_enum_parameter_choices(parameter_schema)
                if parameter_type == ParameterType.ENUM
                else None
            ),
        )

    @staticmethod
    def _set_function_on_parser(parser, task_schema: TaskSchema, ml_func: Callable[..., ResponseBody]):
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
                    case _: # pragma: no cover
                        assert_never(input_schema.input_type)
                inputs[input_schema.key] = input_model
            for parameter_schema in task_schema.parameters:
                parameters[parameter_schema.key] = getattr(args, parameter_schema.key)
            result = ml_func(inputs, parameters)
            return result

        parser.set_defaults(func=func)

    def _add_subparser(self, subparsers, endpoint: EndpointDetails):
        name = self._get_name_of_subcommand(endpoint)
        helpp = endpoint.short_title

        task_schema = endpoint.task_schema_func()

        input_schemas = task_schema.inputs
        parameter_schemas = task_schema.parameters

        subcommand_parser = subparsers.add_parser(name, help=helpp)

        for input_schema in input_schemas:
            self._add_input_argument_to_parser(subcommand_parser, input_schema)
        for parameter_schema in parameter_schemas:
            self._add_parameter_argument_to_parser(subcommand_parser, parameter_schema)
        self._set_function_on_parser(subcommand_parser, task_schema, endpoint.func)
    
    def _setup_cli(self):
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
    
    def _parse_args(self, args: Sequence[str] | None = None):
        parsed_args = self._parser.parse_args(args)
        return parsed_args

    def _run_cli_and_return(self, parsed_args: Namespace) -> Optional[ResponseBody]:
        if parsed_args.func:
            return parsed_args.func(parsed_args)
        raise SystemExit("FATAL: No function defined") # pragma: no cover

    def run_cli(self, args: Sequence[str] | None = None):
        self._setup_cli()
        parsed_args = self._parse_args(args)
        response_body = self._run_cli_and_return(parsed_args)
        if response_body is None:
            raise SystemExit("FATAL: No response body") # pragma: no cover
        print()
        print(json.dumps(json.loads(response_body.model_dump_json()), indent=4))
