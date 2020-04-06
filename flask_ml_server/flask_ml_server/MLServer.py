# -*- coding: utf-8 -*-
"""
    flask_ml.MLServer
    ~~~~~~~~~
    This module implements MLServer object.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""

from flask import Flask, current_app, request, Response
from encoder_decoder import DTypes
from encoder_decoder.default_config import encoders, decoders, extract, wrap
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_response(output):
    return json.dumps(output, cls = NumpyEncoder)

class MLServer(object):
    """The MLServer object is a wrapper class for the flask app object. It
    provides a decorator for turning a machine learning prediction function
    into a WebService on an applet.
    """



    def __init__(self, name):
        """Instantiates the MLServer object as a wrapper for the Flask app.
        Initializes '/' and '/get_available_models' as default rules.
        The landing page blocks machine learning functions from holding the
        default route. The '/get_available_models' returns the prediction
        functions that are being hosted b the server
        """
        self.app = Flask(name)

        @self.app.route("/get_available_models",methods=['GET'])
        def get_models():
            """Returns a list of models as a JSON object
            Format: {"result":['function1','function2',...]}
            """

            # routes that are held for the server
            prebuilt_routes=["/get_available_models","/static/<path:filename>"]
            routes = []
            for rule in self.app.url_map.iter_rules():
                if not str(rule) in prebuilt_routes:
                    routes.append('%s' % str(rule)[1:])

            # return routes as a pickled json object
            response = create_response(routes)
            return Response(response=response)



    def route(self, rule:str, input_type:DTypes, output_type:DTypes=DTypes.STRING):
        '''
        rule : str - the name of the endpoint
        input_type : encoder_decoder.DTypes - the type of data parsing to be used for the input data
        output_type : encoder_decoder.DTypes - the type of data parsing to be used for output of the decorated function
        '''
        if rule is None:
            raise ValueError('The parameter "rule" cannot be None')
        if input_type is None:
            raise ValueError('The parameter "input_type" cannot be None')
        if output_type is None:
            raise ValueError('The parameter "output_type" cannot be None')
        if type(rule) != str:
            raise ValueError('The parameter "rule" is expected to be a string')

        def build_route(ml_function):
            @self.app.route(rule,endpoint=ml_function.__name__,methods=['POST'])
            def prep_ML():
                data_dict = json.loads(request.get_json())
                input_data = decoders[input_type](extract[input_type](data_dict))
                result = ml_function(input_data)
                output = {}
                wrap[output_type](encoders[output_type](result), output)
                output['output_type'] = output_type.value
                response = create_response(output)
                response = Response(response=response, status=200, mimetype="application/json")
                return response
            return prep_ML
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
