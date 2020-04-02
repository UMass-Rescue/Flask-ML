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
from encoder_decoder import DTypes
from encoder_decoder.default_config import decoders, encoders, extract, wrap
import pdb
import warnings
# ==============================================================================


def get_dtype(input):
    if type(input) == str:
        return DTypes.STRING
    elif type(input) == float:
        return DTypes.FLOAT
    elif type(input) == np.ndarray:
        return DTypes.FLOAT_NDARRAY
    else:
        raise ValueError(f'Type of input not supported. Got type as  - {type(input)}')

class MLClient(object):
    """The MLClient object is a wrapper for making requests to an instance
    of the MLServer. It provides useful services for a client to make prediction
    calls to the server and receive outputs without having to format data
    """

    def __init__(self, host=None):
        """Debugging...to be changed

        :param host: specify the host for connection
        """
        if host is None:
            host = 'http://127.0.0.1:5000'
            warnings.warn(f'Host address not specified. Using default host address - {host}')

        self.HOST = host



    def predict(self, input, endpoint):
        """Encode data to input for ML and make post request to ML server.

        :param input: path_to_file to be used as input for model

        :param endpoint: model to use as classification. available
        rules can be found from get_models
        """
        if input is None:
            raise ValueError('The parameter "input" cannot be None')
        if endpoint is None:
            raise ValueError('The parameter "endpoint" cannot be None')

        data = {}
        # pdb.set_trace()
        input_type = get_dtype(input)
        wrap[input_type](encoders[input_type](input), data)
        data = json.dumps(data)
        # Make post request with given endpoint and json data
        response = requests.post(os.path.join(self.HOST, endpoint), json=data)
        response = json.loads(response.text)
        output_type = DTypes(response['output_type'])
        result = decoders[output_type](extract[output_type](response))
        return result



    def get_models(self):
        """Return available models as a list of rule names
        """

        # make get request to get_available_models
        r = requests.get(os.path.join(self.HOST, 'get_available_models'))

        # jsonify output and return
        # format ['function1', 'function2', ...]
        return json.loads(r.text)
