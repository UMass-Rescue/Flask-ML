from flask_ml_client.ml_client import MLClient
import numpy as np

HOST = 'http://127.0.0.1:5000'

client = MLClient(HOST)

models = client.get_models()
print("Models: {}".format(models))

data = np.ones((5,55))
print("Data Input Format: {}".format(type(data)))

classification = models[0]
print("Classification: {}".format(classification))

result = client.predict(data,classification)
print("Result: {}".format(result))
