# -*- coding: utf-8 -*-
"""
    flask_ml.MLClient
    ~~~~~~~~~
    This module implements MLClient object.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""
from pathlib import Path
from fastai.vision import open_image
import aiohttp
import asyncio
import os
import json
import numpy as np
from encoder_decoder import DTypes
from encoder_decoder.default_config import decoders, encoders, extract, wrap
import pdb
import warnings
from functools import partial

def get_dtype(input):
    if type(input) == str:
        return DTypes.STRING
    elif type(input) == float:
        return DTypes.FLOAT
    elif type(input) == np.ndarray:
        return DTypes.FLOAT_NDARRAY
    else:
        raise ValueError(f'Type of input not supported. Got type as  - {type(input)}')

        
# async def post_single(session, endpoint, data):
#     print('posting')
#     async with session.post(endpoint, json=data) as resp:
#         print('reading resp')
#         return await resp.text()


async def post_single(session, endpoint, data):
    print('posting')
    resp = await session.post(endpoint, json=data)
    print('reading resp')
    return await resp.text()
    # async with session.post(endpoint, json=data) as resp:
    #     print('reading resp')
    #     return await resp.text()
    
def read_and_proc_image(path):
    return open_image(path).resize(224).data.numpy()

async def predict_for_path(client, path, endpoint):
#     print('reading image')
    loop = asyncio.get_running_loop()
#     img = await loop.run_in_executor(None, partial(read_and_proc_image, path))
    img = read_and_proc_image(path) # this should be async too
#     print('image read')
    return await client.predict_async(img, endpoint)

class MLClient(object):
    """The MLClient object is a wrapper for making requests to an instance
    of the MLServer. It provides useful services for a client to make prediction
    calls to the server and receive outputs without having to format data
    """

    def __init__(self, session, host:str=None ):
        """
        :param host: specify the host for connection. Must include both the ip address and port number. Ex: 'http://127.0.0.1:5000'
        """
        if host is None:
            host = 'http://127.0.0.1:5000'
            warnings.warn(f'Host address not specified. Using default host address - {host}')

        self.HOST = host
        self.session = session

    def predict(self, input, endpoint:str):
        """Encode data to input for ML and make post request to ML server.

        :param input: path_to_file to be used as input for model

        :param endpoint: model to use as classification. available
        rules can be found from get_models
        """
        res = asyncio.run(self.predict_async(input, endpoint))
        return res

    async def predict_async(self, input, endpoint:str):
        """Encode data to input for ML and make post request to ML server.

        :param input: path_to_file to be used as input for model

        :param endpoint: model to use as classification. available
        rules can be found from get_models
        """
        if input is None:
            raise ValueError('The parameter "input" cannot be None')
        if endpoint is None:
            raise ValueError('The parameter "endpoint" cannot be None')

        data = {}
#         pdb.set_trace()
#         print('encoding data')
        input_type = get_dtype(input)
        wrap[input_type](encoders[input_type](input), data)
        data = json.dumps(data)
        # Make post request with given endpoint and json data
#         response = requests.post(os.path.join(self.HOST, endpoint), json=data)
        print('sending request')
        # response = await post_single(self.session, os.path.join(self.HOST, endpoint), data)
        print('posting')
        resp = await self.session.post(endpoint, json=data)
        print('reading resp')
        response =  await resp.text()

        print('parsing response')
        response = json.loads(response)
        output_type = DTypes(response['output_type'])
        result = decoders[output_type](extract[output_type](response))
        return result

    def get_models(self):
        """Return available models as a list of rule names
        """

        # make get request to get_available_models
        r = requests.get(os.path.join(self.HOST, 'get_available_models'))

        # jsonify output and return
        # format ['function1', 'function2', ...]
        return json.loads(r.text)

async def multi_reqs(num_imgs):    
    async with aiohttp.ClientSession() as session:
        client = MLClient(session, host='http://127.0.0.1:8000')
        imgs = [f for f in Path('../../api-test-harness/data/lfw/').glob('*/*')]

        res = await asyncio.gather(*[ asyncio.ensure_future(predict_for_path(client, p, 'benchmark_imagenet')) for p in imgs[:num_imgs]])
        return res

loop = asyncio.get_event_loop()
task = asyncio.ensure_future(multi_reqs(100))
loop.run_until_complete(task)