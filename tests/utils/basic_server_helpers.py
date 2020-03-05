# -*- coding: utf-8 -*-
"""
    tests.utils.basic_server_helpers
    ~~~~~~~~~
    This module uses MLServer to make a basic server.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""
from torchvision import transforms
import cv2
from PIL import Image
import torch


def object_dection_transform(img):
    """Perform image transformations for alexnet and resnet
    """
    transform = transforms.Compose([
        transforms.Resize(244),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    img = cv2.resize(img, (244,244))
    image = Image.fromarray(img.astype('uint8'), 'RGB')

    image = transform(image)
    image = torch.unsqueeze(image, 0)

    return image

def object_detection_output(out):
    """Format output as a json object with class label and confidence value
    """

    # get labels list externally
    with open('tests/utils/imagenet_classes.txt') as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    result = {"class":imagenet_classes[index[0]], "confidence":percentage[index[0]].item()}
    return result
