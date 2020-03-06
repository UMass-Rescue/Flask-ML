import requests
import numpy as np
from encoder_decoder.dtypes_encode_decode import float_ndarray_to_dict
from encoder_decoder.dtypes_extract_wrap import wrap_data
import os
import json

data = np.ndarray([1,1,1,1,1])
HOST = 'http://127.0.0.1:5000'
endpoint = "img_shape"

print(data.tostring())
b = data.tostring()

print(np.frombuffer(b))

# data = wrap_data(float_ndarray_to_dict(data), {})
#
# print(type(data['data']['str']))
#
# data = json.dumps(data)

# response = requests.post(os.path.join(HOST, endpoint), json=data)
# print(response.text)
