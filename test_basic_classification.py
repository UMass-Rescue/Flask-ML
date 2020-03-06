import requests
import numpy as np
from encoder_decoder.dtypes_encode_decode import float_ndarray_to_dict
from encoder_decoder.dtypes_extract_wrap import wrap_data
import os
import json
# import pdb 

data = np.ones((5,5))
HOST = 'http://127.0.0.1:5000'
endpoint = "img_shape"
# pdb.set_trace()
data = wrap_data(float_ndarray_to_dict(data), {})

# print(type(data['data']['str']))

data = json.dumps(data)

response = requests.post(os.path.join(HOST, endpoint), json=data)
print(response.text)
