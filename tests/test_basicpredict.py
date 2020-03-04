from flask_ml import MLServer
from torchvision import models, transforms
import torch
import requests
import tensorflow as tf
from flask import request, Response
from io import BytesIO
import base64
import cv2
from PIL import Image
import numpy as np
import jsonpickle

serv = MLServer(__name__)

@serv.route('/object_detection', input = {"transform": "tensor"})
def alexnet_object_rec(img):
    alexnet = models.alexnet(pretrained=True)
    batch_t = torch.unsqueeze(img, 0)
    alexnet.eval()
    out = alexnet(batch_t)

    with open('tests/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    result = {"class":classes[index[0]], "confidence":percentage[index[0]].item()}
    return result

@serv.route('/test')
def tester(img):
#   ========= ONLY THING THAT SHOULD BE LEFT IN THIS METHOD  ====
    x = img.shape[1]
    y = img.shape[0]
    result = {"shape":{"x":x,"y":y}}
    return result
#   ========= END MACHINE LEARNING CODE =========================

serv.run()
