from flask_ml import MLServer
from torchvision import models
import torch

serv = MLServer(__name__)


@serv.route('/object_detection_alexnet', input = {"transform": "tensor"}, output={"classification":"imagenet"})
def alexnet_object_rec(img):
    alexnet = models.alexnet(pretrained=True)
    image = torch.unsqueeze(img, 0)
    alexnet.eval()
    out = alexnet(image)
    return out


@serv.route('/object_detection_resnet',input = {"transform": "tensor"}, output={"classification":"imagenet"})
def resnet_object_rec(img):
    resnet = models.resnet101(pretrained=True)
    image = torch.unsqueeze(img, 0)
    resnet.eval()
    out = resnet(image)
    return out

@serv.route('/img_shape')
def tester(img):
    x = img.shape[1]
    y = img.shape[0]
    result = {"shape":{"x":x,"y":y}}
    return result


serv.run()
