from flask import Flask, request, Response
from io_tools import *
from dtypes import InputTypes, OutputTypes
from dtypes_config import encoders, decoders, extract_input, wrap_output


# TODO - Should the methods being passed to self.app.route be defined outside? Would it add a performance overhead to 
# have them be defined everytime a new MLServer is created or everytime a new route is added?

# TODO - what is the return type of this function? Is it even required?
def create_response(data:dict):
    pass

class MLServer(object):

    def __init__(self, app = None):
        self.app = Flask(__name__) if app is None else app

        @self.app.route("/",methods=['GET'])
        def landing():
            # TODO do we need a landing page?
            pass

        @self.app.route("/get_available_models",methods=['GET'])
        def get_models():
            pass

    def route(self, rule, input_type:InputTypes, output_type:OutputTypes=OutputTypes.STRING):
        def build_route(ml_function):
            # TODO - what is the difference between "rule" and "endpoint"? Are we forcing the users to have the endpoint as their function name?
            # is that okay to do?
            @self.app.route(rule,endpoint=ml_function.__name__,methods=['POST'])
            def prep_ML():
                input_data = decoders[input_type](extract_input[input_type](request))
                result = ml_function(input_data)
                output = {}
                wrap_output[output_type](encoders[output_type](result), output) # TODO Any problem with inplace append to dict?
                response = create_response(output)
                response = Response(response=response, status=200, mimetype="application/json")
                return response
            return prep_ML
        return build_route

    def run(self):
        self.app.run()


# import requests
# import base64
# import os
# import json

# class MLClient(object):
#     def __init__(self, HOST=None):
#         if HOST:
#             self.HOST = HOST
#         else:
#             self.HOST = 'http://127.0.0.1:5000'

#     def predict(self, input, endpoint, dtype):
#         """ post image and return the response """
#         if dtype == "single image":
#             img = open(input, 'rb').read()
#             img = base64.b64encode(img).decode('utf-8')
#             data={"type":"single_image","name":input, "image":img}
#         elif dtype == "single text":
#             data={"type":"single_text","text":input}
#         return self.post_predict(data, endpoint)

#     def post_predict(self, data, endpoint):
#         response = requests.post(os.path.join(self.HOST, endpoint), json=data)
#         return json.loads(response.text)

#     def get_models(self):
#         r = requests.get(os.path.join(self.HOST, 'get_available_models'))
#         return json.loads(r.text)["result"]
