import json
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from flask_ml.flask_ml_server.errors import BadRequestError

logger = getLogger(__name__)

from flask import Flask, Response, jsonify, request
from pydantic import ValidationError

from flask_ml.flask_ml_server.models import (
    APIRoutes,
    NoSchemaAPIRoute,
    ResponseBody,
    SchemaAPIRoute,
    TaskSchema,
)
from flask_ml.flask_ml_server.utils import (
    ensure_ml_func_hinting_and_task_schemas_are_valid,
    ensure_ml_func_parameters_are_typed_dict,
    no_schema_get_inputs,
    no_schema_get_parameters,
    schema_get_inputs,
    schema_get_parameters,
    schema_get_sample_payload,
    type_hinting_get_sample_payload,
    validate_data_has_keys,
)


@dataclass
class EndpointDetailsNoSchema:
    rule: str
    payload_schema_rule: str
    sample_payload_rule: str
    func: Callable[..., ResponseBody]


@dataclass
class EndpointDetails(EndpointDetailsNoSchema):
    task_schema_rule: str
    task_schema_func: Callable[[], TaskSchema]
    short_title: str
    order: int


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
        self.endpoints: List[EndpointDetailsNoSchema] = []

        @self.app.route("/api/routes", methods=["GET"])
        def list_routes():
            """
            Lists all the routes/endpoints available in the Flask app.
            """
            routes = [
                (
                    SchemaAPIRoute(
                        task_schema=endpoint.task_schema_rule,
                        run_task=endpoint.rule,
                        sample_payload=endpoint.sample_payload_rule,
                        payload_schema=endpoint.payload_schema_rule,
                        short_title=endpoint.short_title,
                        order=endpoint.order,
                    )
                    if isinstance(endpoint, EndpointDetails)
                    else NoSchemaAPIRoute(
                        run_task=endpoint.rule,
                        sample_payload=endpoint.sample_payload_rule,
                        payload_schema=endpoint.payload_schema_rule,
                    )
                )
                for endpoint in self.endpoints
            ]
            return jsonify(APIRoutes(root=routes).model_dump(mode="json"))

    def route(
        self,
        rule: str,
        task_schema_func: Optional[Callable[[], TaskSchema]] = None,
        short_title: Optional[str] = None,
        order: int = 0,
    ):
        """
        rule : str - the name of the endpoint
        input_type : str - the type of the input data
        """

        def build_route(ml_function: Callable[[Any, Any], ResponseBody]):
            ensure_ml_func_parameters_are_typed_dict(ml_function)
            if task_schema_func is not None:
                ensure_ml_func_hinting_and_task_schemas_are_valid(ml_function, task_schema_func())
                endpoint = EndpointDetails(
                    rule=rule,
                    task_schema_rule=rule + "/task_schema",
                    sample_payload_rule=rule + "/sample_payload",
                    payload_schema_rule=rule + "/payload_schema",
                    func=ml_function,
                    task_schema_func=task_schema_func,
                    short_title=short_title or "",
                    order=order,
                )
                self.endpoints.append(endpoint)

                @self.app.route(
                    endpoint.task_schema_rule, endpoint=endpoint.task_schema_rule, methods=["GET"]
                )
                def get_task_schema():
                    return jsonify(endpoint.task_schema_func().model_dump(mode="json"))

                @self.app.route(
                    endpoint.sample_payload_rule, endpoint=endpoint.sample_payload_rule, methods=["GET"]
                )
                def get_sample_payload():
                    return jsonify(
                        schema_get_sample_payload(endpoint.task_schema_func()).model_dump(mode="json")
                    )

                @self.app.route(
                    endpoint.payload_schema_rule, endpoint=endpoint.payload_schema_rule, methods=["GET"]
                )
                def get_payload_schema():
                    return jsonify(schema_get_sample_payload(endpoint.task_schema_func()).model_json_schema())

                @self.app.route(rule, endpoint=ml_function.__name__, methods=["POST"])
                def wrapper():
                    try:
                        data = request.get_json()
                        validate_data_has_keys(data, ["inputs", "parameters"])
                        json_inputs = data["inputs"]
                        json_parameters = data["parameters"]

                        assert task_schema_func is not None, "FATAL: Input schema cannot be None here"
                        task_schema = task_schema_func()
                        ensure_ml_func_hinting_and_task_schemas_are_valid(ml_function, task_schema)

                        inputs = schema_get_inputs(task_schema, json_inputs)
                        parameters: Dict[str, Union[str, int, float]] = schema_get_parameters(
                            task_schema, json_parameters
                        )
                        result = ml_function(inputs, parameters)
                        logger.info(f"200: Successful request")
                        response = Response(
                            status=200, mimetype="application/json", response=result.model_dump_json()
                        )
                    except ValidationError as e:
                        error = {"error": e.errors(), "status": "VALIDATION_ERROR"}
                        logger.error(f"400: Validation error: {error}")
                        response = Response(
                            status=400, mimetype="application/json", response=json.dumps(error)
                        )
                    except BadRequestError as e:
                        logger.error(f"400: Bad request: {e}")
                        response = Response(
                            status=400,
                            mimetype="application/json",
                            response=json.dumps({"error": str(e), "status": "VALIDATION_ERROR"}),
                        )
                    except Exception as e:
                        logger.error(f"500: Internal server error: {repr(e)}")
                        logger.error(traceback.format_exc())
                        response = Response(
                            status=500,
                            mimetype="application/json",
                            response=json.dumps({"error": repr(e), "status": "SERVER_ERROR"}),
                        )
                    return response

                return wrapper
            else:
                endpoint = EndpointDetailsNoSchema(
                    rule=rule,
                    payload_schema_rule=rule + "/payload_schema",
                    sample_payload_rule=rule + "/sample_payload",
                    func=ml_function,
                )
                self.endpoints.append(endpoint)
                hints = get_type_hints(ml_function)

                @self.app.route(
                    endpoint.sample_payload_rule, endpoint=endpoint.sample_payload_rule, methods=["GET"]
                )
                def get_sample_payload():
                    return jsonify(type_hinting_get_sample_payload(hints).model_dump(mode="json"))

                @self.app.route(
                    endpoint.payload_schema_rule, endpoint=endpoint.payload_schema_rule, methods=["GET"]
                )
                def get_payload_schema():
                    return jsonify(type_hinting_get_sample_payload(hints).model_json_schema())

                @self.app.route(rule, endpoint=ml_function.__name__, methods=["POST"])
                def wrapper():
                    try:
                        data = request.get_json()
                        validate_data_has_keys(data, ["inputs", "parameters"])
                        json_inputs = data["inputs"]
                        json_parameters = data["parameters"]

                        inputs = no_schema_get_inputs(get_type_hints(hints["inputs"]), json_inputs)
                        parameters: Dict[str, Union[str, int, float]] = no_schema_get_parameters(
                            get_type_hints(hints["parameters"]), json_parameters
                        )
                        result = ml_function(inputs, parameters)
                        logger.info(f"200: Successful request")
                        response = Response(
                            status=200, mimetype="application/json", response=result.model_dump_json()
                        )
                    except ValidationError as e:
                        error = {"error": e.errors(), "status": "VALIDATION_ERROR"}
                        logger.error(f"400: Validation error: {error}")
                        response = Response(
                            status=400, mimetype="application/json", response=json.dumps(error)
                        )
                    except BadRequestError as e:
                        logger.error(f"400: Bad request: {e}")
                        response = Response(
                            status=400,
                            mimetype="application/json",
                            response=json.dumps({"error": str(e), "status": "VALIDATION_ERROR"}),
                        )
                    except Exception as e:
                        logger.error(f"500: Internal server error: {repr(e)}")
                        logger.error(traceback.format_exc())
                        response = Response(
                            status=500,
                            mimetype="application/json",
                            response=json.dumps({"error": repr(e), "status": "SERVER_ERROR"}),
                        )
                    return response

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
