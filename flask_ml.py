from flask import Flask, request, Response
from torchvision import transforms
import jsonpickle
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def readb64(base64_string, size = None):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    if size:
        return cv2.resize(img, size)
    return img

def transfunc(bytes):
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])
    img = readb64(bytes)
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    return transform(image)


def prepare_data(input, data):
    type = data["type"]

    if type == "single_image":
        bytes = data["image"]
        if (input == {}):
            img = readb64(bytes, (244,244))
        elif input["transform"] == "tensor":
            img = transfunc(bytes)
        return img
    else:
        return "invalid type"

def return_response(output,result):
    output["result"] = result
    response = output
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


class MLServer(object):

    def __init__(self, app = None):
        self.app = Flask(__name__)

    def route(self, rule, input = {}, output = {}):
        def build_route(ML_Function):
            @self.app.route(rule,endpoint=ML_Function.__name__,methods=['POST'])
            def prep_ML():
                data = request.get_json()
                ml_input = prepare_data(input, data)
                result = ML_Function(ml_input)
                response = return_response(output, result)
                return response
            return prep_ML
        return build_route


    def run(self):
        self.app.run()
