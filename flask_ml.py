from flask import Flask, request, Response
from io_tools import *

class MLServer(object):

    def __init__(self, app = None):
        self.app = Flask(__name__)

        @self.app.route("/",methods=['GET'])
        def landing():
            return "ML-Server Landing Page"


        @self.app.route("/get_available_models",methods=['GET'])
        def get_models():
            prebuilt_routes = ["/get_available_models","/static/<path:filename>", "/"]
            routes = []
            for rule in self.app.url_map.iter_rules():
                if not str(rule) in prebuilt_routes:
                    routes.append('%s' % str(rule)[1:])
            response = return_response({},routes)
            return Response(response=response)



    def route(self, rule, input = {}, output = {"classification":"miscellaneous"}):
        def build_route(ML_Function):
            @self.app.route(rule,endpoint=ML_Function.__name__,methods=['POST'])
            def prep_ML():
                data = request.get_json()
                ml_input = prepare_data(input, data)
                result = ML_Function(ml_input)
                output["model"] = str(rule)[1:]
                if not output["classification"] == "miscellaneous":
                    result = wrap_result(output, result)
                response = return_response(output, result)
                response = Response(response=response, status=200, mimetype="application/json")
                return response
            return prep_ML
        return build_route

    def run(self):
        self.app.run()


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
        """ post image and return the response """
        if dtype == "single image":
            img = open(input, 'rb').read()
            img = base64.b64encode(img).decode('utf-8')
            data={"type":"single_image","name":input, "image":img}
        elif dtype == "single text":
            data={"type":"single_text","text":input}
        return self.post_predict(data, endpoint)

    def post_predict(self, data, endpoint):
        response = requests.post(os.path.join(self.HOST, endpoint), json=data)
        return json.loads(response.text)

    def get_models(self):
        r = requests.get(os.path.join(self.HOST, 'get_available_models'))
        return json.loads(r.text)["result"]
