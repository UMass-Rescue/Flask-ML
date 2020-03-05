import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import jsonpickle


def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    return img


def prepare_data(input, data):
    if data["type"] == "single_image":
        bytes = data["image"]
        img = readb64(bytes)
        return img
    else:
        return "invalid type"

def return_response(output,result):
    output["result"] = result
    response_pickled = jsonpickle.encode(output)
    return response_pickled
