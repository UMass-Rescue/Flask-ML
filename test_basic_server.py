# -*- coding: utf-8 -*-
"""
    tests.test_basic_server
    ~~~~~~~~~
    This module uses MLServer to make a basic server.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""

# import flask_ml
from flask_ml_server.ml_server import MLServer
from encoder_decoder import DTypes
# import pdb
# make a server instance
serv = MLServer(__name__)

@serv.route('/img_shape', input_type=DTypes.FLOAT_NDARRAY, output_type = DTypes.STRING)
def tester(img):
    """Return image dimensions
    """
    # pdb.set_trace()
    x = img.shape[1]
    y = img.shape[0]
    result = "x = {}, y = {}".format(x,y)
    return result

# begin server instance
serv.run()