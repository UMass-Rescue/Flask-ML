import inspect
from typing import get_args, get_origin, get_type_hints

from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError

from .models import ErrorResponseModel, RequestModel, ResponseModel


def get_first_param_name(func):
    sig = inspect.signature(func)
    params = sig.parameters.values()
    if len(params) == 0:
        return None
    return next(iter(params)).name


def format_schema_output(input_schema, output_schema):
    return {"inputs": input_schema, "output": output_schema}


def get_function_schema(fn):
    first_param_name = get_first_param_name(fn)
    if not first_param_name:
        return format_schema_output(None, None)
    type_hints = get_type_hints(fn)
    input_type = type_hints.get(first_param_name)
    if get_origin(input_type) is list:
        list_args = get_args(input_type)
        if len(list_args) == 0:
            return format_schema_output(None, None)
        inner_type = list_args[0]
        if issubclass(inner_type, BaseModel):
            input_schema = {"type": "array", "items": inner_type.model_json_schema()}
        else:
            input_schema = None
    else:
        input_schema = None

    output_model = type_hints.get("return")
    if output_model and issubclass(output_model, BaseModel):
        output_schema = output_model.model_json_schema()
    else:
        output_schema = None

    return format_schema_output(input_schema, output_schema)


class MLServer(object):
    """
    The MLServer object is a wrapper class for the flask app object. It
    provides a decorator for turning a machine learning prediction function
    into a WebService on an applet.
    """

    def __init__(self, name):
        """
        Instantiates the MLServer object as a wrapper for the Flask app.
        """
        self.app = Flask(name, static_folder=None)
        self.endpoint2function = {}

        @self.app.route("/api/routes", methods=["GET"])
        def list_routes():
            """
            Lists all the routes/endpoints available in the Flask app.
            """
            routes = []
            for rule in self.app.url_map.iter_rules():
                schema = (
                    None
                    if rule.rule not in self.endpoint2function
                    else get_function_schema(self.endpoint2function[rule.rule])
                )
                route_info = {
                    "rule": rule.rule,
                    "methods": list(rule.methods - {"HEAD", "OPTIONS"}),
                    "schema": schema,
                }
                routes.append(route_info)
            return jsonify(routes)

    def route(self, rule: str, input_type: str):
        """
        rule : str - the name of the endpoint
        input_type : str - the type of the input data
        """
        if rule is None:
            raise ValueError('The parameter "rule" cannot be None')
        if input_type is None:
            raise ValueError('The parameter "input_type" cannot be None')
        if type(rule) != str:
            raise ValueError('The parameter "rule" is expected to be a string')

        def build_route(ml_function):
            self.endpoint2function[rule] = ml_function

            @self.app.route(rule, endpoint=ml_function.__name__, methods=["POST"])
            def wrapper():
                try:
                    data = request.get_json()
                    data = RequestModel(**data)
                    result = ml_function(data.inputs, data.parameters)
                    response = ResponseModel(status="SUCCESS", results=result)
                    return response.get_response()
                except ValidationError as e:
                    error_details = e.errors()
                    error_details = [
                        {
                            "type": err.get("type", ""),
                            "input": err.get("input", ""),
                            "msg": err.get("msg", ""),
                        }
                        for err in error_details
                    ]
                    return ErrorResponseModel(
                        status="VALIDATION_ERROR", errors=error_details
                    ).get_response()

            return wrapper

        return build_route

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        """Runs the application on a local development server.

        Do not use ``run()`` in a production setting. It is not intended to
        meet security and performance requirements for a production server.
        Instead, see :ref:`deployment` for WSGI server recommendations.


        If the :attr:`debug` flag is set the server will automatically reload
        for code changes and show a debugger in case an exception happened.


        If you want to run the application in debug mode, but disable the
        code execution on the interactive debugger, you can pass
        ``use_evalex=False`` as parameter.  This will keep the debugger's
        traceback screen active, but disable code execution.


        :param host: the hostname to listen on. Set this to ``'0.0.0.0'`` to
            have the server available externally as well. Defaults to
            ``'127.0.0.1'`` or the host in the ``SERVER_NAME`` config variable
            if present.
        :param port: the port of the webserver. Defaults to ``5000`` or the
            port defined in the ``SERVER_NAME`` config variable if present.
        :param debug: if given, enable or disable debug mode. See
            :attr:`debug`.
        :param load_dotenv: Load the nearest :file:`.env` and :file:`.flaskenv`
            files to set environment variables. Will also change the working
            directory to the directory containing the first file found.
        :param options: the options to be forwarded to the underlying Werkzeug
            server. See :func:`werkzeug.serving.run_simple` for more
            information.
        """
        self.app.run(host, port, debug, load_dotenv, **options)
