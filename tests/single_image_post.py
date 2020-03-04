import requests
import base64
import os

HOST = 'http://127.0.0.1:5000'


def post_image(img_file, endpoint):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    img = base64.b64encode(img).decode('utf-8')

    data={"type":"single_image","name":img_file, "image":img}
    response = requests.post(os.path.join(HOST, endpoint), json=data)
    return response

r = requests.get(os.path.join(HOST, 'get_models'))
print(r.text)

r = post_image("tests/dog.jpg","img_shape")
print(r.text)

r = post_image("tests/dog.jpg","object_detection_alexnet")
print(r.text)

r = post_image("tests/dog.jpg","object_detection_resnet")
print(r.text)
