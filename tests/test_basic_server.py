# -*- coding: utf-8 -*-
"""
    tests.test_basic_server
    ~~~~~~~~~
    This module uses MLServer to make a basic server.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""

# import flask_ml
from flask_ml import MLServer

# make a server instance
serv = MLServer(__name__)

@serv.route('/img_shape')
def tester(img):
    """Return image dimensions
    """
    x = img.shape[1]
    y = img.shape[0]
    result = {"shape":{"x":x,"y":y}}
    return result

# begin server instance
serv.run()
