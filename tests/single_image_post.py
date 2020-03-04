import requests
import base64
import time
import os

HOST = 'http://127.0.0.1:5000'


def post_image(img_file, endpoint):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    img = base64.b64encode(img).decode('utf-8')

    data={"type":"single_image","name":img_file, "image":img}
    start_time = time.time()
    response = requests.post(os.path.join(HOST, endpoint), json=data)
    print("Post Image Response Time: {}".format(time.time() - start_time))
    return response


r = post_image("tests/dog.jpeg","test")
print(r.text)


r = post_image("tests/dog.jpeg","object_detection")
print(r.text)
