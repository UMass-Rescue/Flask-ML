# -*- coding: utf-8 -*-
"""
    flask_ml.MLClient
    ~~~~~~~~~
    This module implements MLClient object.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""


# CHANGE
# ==============================================================================
import requests
import os
import json
import numpy as np
from encoder_decoder.dtypes import InputTypes, OutputTypes
from encoder_decoder.default_config import decoders, encoders, extract_input, wrap_output
import pdb
# ==============================================================================


def get_dtype(input):
    if type(input) == str:
        return OutputTypes.STRING
    elif type(input) == float:
        return OutputTypes.FLOAT
    elif type(input) == np.ndarray:
        return OutputTypes.FLOAT_NDARRAY
    else:
        raise ValueError(f'Type of input not supported. Got type as  - {type(input)}')

class MLClient(object):
    """The MLClient object is a wrapper for making requests to an instance
    of the MLServer. It provides useful services for a client to make prediction
    calls to the server and receive outputs without having to format data
    """

    def __init__(self, Host ='http://127.0.0.1:5000'):
        """Debugging...to be changed

        :param HOST: specify the Host for connection
        :param Port: specify the port for connection
        """
        self.HOST = Host



    def predict(self, input, endpoint):
        """Encode data to input for ML and make post request to ML server.

        :param input: path_to_file to be used as input for model

        :param endpoint: model to use as classification. available
        rules can be found from get_models
        
        """
# CHANGE
# ==============================================================================
        data = {}
        # pdb.set_trace()
        input_type = get_dtype(input)
        wrap_output[input_type](encoders[input_type](input), data)
        data = json.dumps(data)
# ==============================================================================
        # Make post request with given endpoint and json data
        response = requests.post(os.path.join(self.HOST, endpoint), json=data)
        response = json.loads(response.text)
        output_type = InputTypes(response['output_type'])
        result = decoders[output_type](extract_input[output_type](response))
        return result



    def get_models(self):
        """Return available models as a dictionary of rule names
        """

        # make get request to get_available_models
        r = requests.get(os.path.join(self.HOST, 'get_available_models'))

        # jsonify output and return
        # format {"result":['function1', 'function2', ...]}
        return json.loads(r.text)
