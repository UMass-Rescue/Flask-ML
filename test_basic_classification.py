from flask_ml_client.ml_client import MLClient
import numpy as np


HOST = 'http://127.0.0.1:5000'

client = MLClient(HOST)

models = client.get_models()
data = np.ones((5,55))
classification = 'img_shape'
result = client.predict(data,classification)
print("Models: {}".format(models))
print("Data: {}".format(data))
print("Classification: {}".format(classification))
print("Result: {}".format(result))
