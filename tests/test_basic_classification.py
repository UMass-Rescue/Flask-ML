# -*- coding: utf-8 -*-
"""
    tests.test_basic_classification
    ~~~~~~~~~
    This module uses MLClient to make server calls to the basic_server.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""
# import MLClient
from flask_ml import MLClient

# make a client instance
clie = MLClient()

# print available models json object
models = clie.get_models()
print(models)

result = clie.predict("tests/utils/dog.jpg","img_shape", "single image")
print(result)
