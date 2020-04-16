# -*- coding: utf-8 -*-
"""
    flask_ml.MLServer
    ~~~~~~~~~
    This module implements MLServer object.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""

# from flask import Flask, current_app, request, Response
from fastapi import FastAPI, Response, Request
from encoder_decoder import DTypes
from encoder_decoder.default_config import encoders, decoders, extract, wrap
import json
import numpy as np
import pdb

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

    def __init__(self, name=None):
        """Instantiates the MLServer object as a wrapper for the Flask app.
        Initializes '/' and '/get_available_models' as default rules.
        The landing page blocks machine learning functions from holding the
        default route. The '/get_available_models' returns the prediction
        functions that are being hosted b the server
        """
        # TODO title=name? is that even required?
        self.app = FastAPI()

        # @self.app.get("/get_available_models")
        # def get_models():
        #     """Returns a list of models as a JSON object
        #     Format: {"result":['function1','function2',...]}
        #     """

        #     # routes that are held for the server
        #     prebuilt_routes=["/get_available_models","/static/<path:filename>"]
        #     routes = []
        #     for rule in self.app.url_map.iter_rules():
        #         if not str(rule) in prebuilt_routes:
        #             routes.append('%s' % str(rule)[1:])

        #     # return routes as a pickled json object
        #     response = create_response(routes)
        #     return response



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
            @self.app.post(rule)
            async def prep_ML(request:Request):
                body = await request.json()
                data_dict = json.loads(body)
                input_data = decoders[input_type](extract[input_type](data_dict))
                result = ml_function(input_data)
                output = {}
                wrap[output_type](encoders[output_type](result), output)
                output['output_type'] = output_type.value
                response = create_response(output)
                response = Response(content=response, status_code=200, media_type="application/json")
                return response
            return prep_ML
        return build_route

