# Flask-ML
![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)

## MLServer
The MLServer object is a wrapper class for the flask app object. It provides a decorator for turning a machine learning prediction function into a WebService on an applet.

### Usage
```Python3
from flask_ml import MLServer

# import machine learning models
from torchvision import models

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

serv.run()
```

## MLClient

The MLClient object is a wrapper for making requests to an instance of the MLServer. It provides useful services for a client to make prediction calls to the server and receive outputs without having to format data



## Authors
Jagath Jai Kumar
Prasanna Lakkur Subramanyam



This project is licensed under the MIT License - see the LICENSE file for details
