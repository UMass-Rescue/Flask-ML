import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import jsonpickle
from torchvision import transforms
import torch


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
    transforms.Resize(244),
    #transforms.CenterCrop(224),
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
    response_pickled = jsonpickle.encode(output)
    return response_pickled
    


def wrap_result(output, result):
    if output["classification"] == "imagenet":
        with open('tests/imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        _, index = torch.max(result, 1)
        percentage = torch.nn.functional.softmax(result, dim=1)[0] * 100
        result = {"class":classes[index[0]], "confidence":percentage[index[0]].item()}
        return result
