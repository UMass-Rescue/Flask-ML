# -*- coding: utf-8 -*-
"""
    tests.test_torch_classification
    ~~~~~~~~~
    This module uses MLClient to make server calls to the torch_server.
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

result = clie.predict("tests/utils/dog.jpg","object_detection_alexnet", "single image")
print(result)

result = clie.predict("tests/utils/dog.jpg","object_detection_resnet", "single image")
print(result)
