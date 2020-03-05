from flask import Flask, current_app, request, Response
from .io_tools import *

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
            @self.app.route(rule, endpoint=ML_Function.__name__, methods=['POST'])
            def prep_ML():

                data = request.get_json()
                ml_input = prepare_data(input, data)

                result = ML_Function(ml_input)

                output["model"] = str(rule)[1:]
                response = return_response(output, result)
                response = Response(response=response, status=200, mimetype="application/json")

                return response
            return prep_ML
        return build_route

    def run(self):
        self.app.run()
