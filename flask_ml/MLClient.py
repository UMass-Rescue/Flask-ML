import requests
import base64
import os
import json

class MLClient(object):
    def __init__(self, HOST=None):
        if HOST:
            self.HOST = HOST
        else:
            self.HOST = 'http://127.0.0.1:5000'

    def predict(self, input, endpoint, dtype):
        if dtype == "single image":
            img = open(input, 'rb').read()
            img = base64.b64encode(img).decode('utf-8')
            data={"type":"single_image","name":input, "image":img}
        elif dtype == "single text":
            data={"type":"single_text","text":input}
        response = requests.post(os.path.join(self.HOST, endpoint), json=data)
        return json.loads(response.text)

    def get_models(self):
        r = requests.get(os.path.join(self.HOST, 'get_available_models'))
        return json.loads(r.text)["result"]
