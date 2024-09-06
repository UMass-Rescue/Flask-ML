from flask import Flask, current_app, request, Response
import json
from .response import ErrorResponse

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
        self.app = Flask(name)


    def route(self, rule: str, input_type: str):
        '''
        rule : str - the name of the endpoint
        input_type : str - the type of the input data
        '''
        if rule is None:
            raise ValueError('The parameter "rule" cannot be None')
        if input_type is None:
            raise ValueError('The parameter "input_type" cannot be None')
        if type(rule) != str:
            raise ValueError('The parameter "rule" is expected to be a string')

        def build_route(ml_function):
            @self.app.route(rule, endpoint=ml_function.__name__, methods=['POST'])
            def wrapper():
                data = request.get_json()
                if 'data_type' not in data:
                    return ErrorResponse('The input data must contain a "data_type" field', status=400).get_response()
                if data['data_type'] != input_type:
                    return ErrorResponse(f'The input data type must be {input_type}', status=400).get_response()
                if 'inputs' not in data:
                    return ErrorResponse('The input data must contain an "inputs" field', status=400).get_response()
                inputs = data['inputs']
                parameters = data['parameters'] if 'parameters' in data else {}
                return ml_function(inputs, parameters)
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
