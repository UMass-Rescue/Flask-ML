# -*- coding: utf-8 -*-
"""
    tests.test_torch_server
    ~~~~~~~~~
    This module uses MLServer to make a torch server.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""

# import flask_ml
from flask_ml import MLServer

# import machine learning models
from torchvision import models

# import custom image transforms and output formats
from utils.basic_server_helpers import *

# make a server instance
serv = MLServer(__name__)

@serv.route('/object_detection_alexnet', input = {}, output={"classification":"imagenet"})
def alexnet_object_rec(img):
    """Run alexnet object detection on image, return image label and confidence
    """
    image = object_dection_transform(img)
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    out = alexnet(image)
    return object_detection_output(out)


@serv.route('/object_detection_resnet',input = {}, output={"classification":"imagenet"})
def resnet_object_rec(img):
    """Run resnet object detection on image, return image label and confidence
    """

    image = object_dection_transform(img)
    resnet = models.resnet101(pretrained=True)
    resnet.eval()
    out = resnet(image)
    return object_detection_output(out)

# begin server instance
serv.run()
