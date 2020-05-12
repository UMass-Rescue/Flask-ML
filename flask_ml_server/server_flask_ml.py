from flask_ml_server import MLServer
from encoder_decoder import DTypes
from ml_models import ImageNetClassifier

server = MLServer(__name__)

clf = ImageNetClassifier()

@server.route('/benchmark_imagenet', input_type=DTypes.FLOAT_NDARRAY, output_type=DTypes.STRING)
def predict(img):
    print('Received request')
    ret = clf.predict(img)
    # print(ret)
    return ret


# server.run(host='localhost', port=13039)

app = server.app