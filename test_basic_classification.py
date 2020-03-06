import requests
import numpy as np
from encoder_decoder.dtypes_encode_decode import float_ndarray_to_dict
from encoder_decoder.dtypes_extract_wrap import wrap_data
import os
import json
from flask_ml_client.ml_client import MLClient
# import pdb 
HOST = 'http://127.0.0.1:5000'
# data = np.ones((5,5))
# endpoint = "img_shape"
# # pdb.set_trace()
# data = wrap_data(float_ndarray_to_dict(data), {})

# # print(type(data['data']['str']))

# data = json.dumps(data)

# response = requests.post(os.path.join(HOST, endpoint), json=data)
# print(response.text)

client = MLClient(HOST)
print(client.predict(np.ones((5,215)), 'img_shape'))
