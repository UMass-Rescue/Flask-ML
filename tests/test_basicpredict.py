from flask_ml import MLServer
from torchvision import models, transforms
import torch
import cv2
from PIL import Image

serv = MLServer(__name__)


def object_dection_transform(img):
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
    with open('tests/imagenet_classes.txt') as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    result = {"class":imagenet_classes[index[0]], "confidence":percentage[index[0]].item()}
    return result

@serv.route('/object_detection_alexnet', input = {}, output={"classification":"imagenet"})
def alexnet_object_rec(img):
    image = object_dection_transform(img)
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    out = alexnet(image)
    return object_detection_output(out)


@serv.route('/object_detection_resnet',input = {}, output={"classification":"imagenet"})
def resnet_object_rec(img):
    image = object_dection_transform(img)
    resnet = models.resnet101(pretrained=True)
    resnet.eval()
    out = resnet(image)
    return object_detection_output(out)

@serv.route('/img_shape')
def tester(img):
    x = img.shape[1]
    y = img.shape[0]
    result = {"shape":{"x":x,"y":y}}
    return result


serv.run()
