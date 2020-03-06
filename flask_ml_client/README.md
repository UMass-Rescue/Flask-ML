# Flask-ML
## MLClient

The MLClient object is a wrapper for making requests to an instance of the MLServer. It provides useful services for a client to make prediction calls to the server and receive outputs without having to format data

### Usage
```Python3
# import MLClient
from flask_ml import MLClient

# make a client instance
clie = MLClient()

# print available models json object
models = clie.get_models()
print(models)

# clie.predict will make a post request to the server and return the result
img_arr_np = read_image_in_numpy_arr("tests/utils/dog.jpg")
result = clie.predict(img_arr_np, "object_detection_alexnet")
print(result)
```
