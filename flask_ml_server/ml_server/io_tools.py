# -*- coding: utf-8 -*-
'''
    flask_ml.io_tools
    -----------------
    General utilities for reading encoded data and outputting
    pickled json files for responses
'''

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import jsonpickle

def decode_data(input_format, data):
    """Prepare input for machine learning. Image base64 strings will be
    converted to cv2 ndarray,

    Not Supported: text will be stripped of whitespace and newlines

    :param input_format: input formating schema
    :param data: data to be preprocessed
    """
    if data["type"] == "single_image":

        # extract bytes from data packet
        bytes = data["image"]

        # convert to image and return
        img = readb64(bytes)
        return img
    else:

        # invalid type exception
        return "invalid type"

def return_response(output,result):
    """Return machine learning output as a pickled json

    :param output: dictionary to be populated with result and other formatting
    data
    :param result: machine learning result, should be formatted as a json object
    """

    # insert result into output dictionary
    output["result"] = result

    # pickle dictionary
    response_pickled = jsonpickle.encode(output)
    return response_pickled


def readb64(base64_string):
    """This will decode a base64 string and convert to a cv2 image

    :param base64_string: string to be decoded and returned as image
    """

    # Use BytesIO to decode input string
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))

    # Use Pillow to convert to image
    pimg = Image.open(sbuf)

    # OpenCV to colorize image
    img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    return img
