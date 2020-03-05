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
import base64
import os
import json
# ==============================================================================



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



    def predict(self, input, endpoint, dtype):
        """Encode data to input for ML and make post request to ML server.

        :param input: path_to_file to be used as input for model

        :param endpoint: model to use as classification. available
        rules can be found from get_models

        :param dtype: type of data to be classified.
        ["single image", "single_text"]
        """
# CHANGE
# ==============================================================================
        if dtype == "single image":
            # open image and base64 encode
            img = open(input, 'rb').read()
            img = base64.b64encode(img).decode('utf-8')

            # json format for post request
            data={"type":"single_image","name":input, "image":img}

        elif dtype == "single text":
            # NOT currently supported
            data={"type":"single_text","text":input}
# ==============================================================================
        # Make post request with given endpoint and json data
        response = requests.post(os.path.join(self.HOST, endpoint), json=data)
        return json.loads(response.text)



    def get_models(self):
        """Return available models as a dictionary of rule names
        """

        # make get request to get_available_models
        r = requests.get(os.path.join(self.HOST, 'get_available_models'))

        # jsonify output and return
        # format {"result":['function1', 'function2', ...]}
        return json.loads(r.text)
