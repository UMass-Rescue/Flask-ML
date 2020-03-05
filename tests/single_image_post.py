from flask_ml import MLClient

HOST = 'http://127.0.0.1:5000'

clie = MLClient(HOST)

models = clie.get_models()
print(models)

result = clie.predict("tests/dog.jpg","img_shape", "single image")
print(result)

result = clie.predict("tests/dog.jpg","object_detection_alexnet", "single image")
print(result)

result = clie.predict("tests/dog.jpg","object_detection_resnet", "single image")
print(result)
