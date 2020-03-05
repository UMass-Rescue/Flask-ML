# class InputTypes(Enum):
#     IMAGE = auto() # 3d numpy array
#     IMAGE_BATCH = auto() # 4d numpy array
#     STRING = auto() # python string
#     FLOAT_NDARRAY = auto() # numpy ndarray
#     FLOAT = auto() # python float or numpy float. How about anything that converts to float using "float" method? TODO

# class OutputTypes(Enum):
#     STRING = auto() # python string
#     IMAGE = auto() # 3d numpy array
#     FLOAT_NDARRAY = auto() # numpy ndarray
#     FLOAT = auto() # python float or numpy float. How about anything that converts to float using "float" method? TODO

'''
We send JSON and receive JSON in our network requests. We can have specific fields in JSON pointing to encoded types as specified below

Image (3d numpy array) -> Base64 Encoded String
Base64 Encoded String -> Image (3d numpy array)

Image Batch (4d numpy array) -> list of Base64 Encoded Strings? TODO
list of Base64 Encoded Strings?  -> Image Batch (4d numpy array)

FLOAT_NDARRAY (numpy ndarray) -> string? (np.tostring)
string -> FLOAT_NDARRAY (numpy ndarray)

Float -> No Encoding? or np.tostring if numpy array? TODO

String -> No Encoding? 


TODO Concern with numpy array conversion - precision shouldn't be lost. float64, for example, should have all the 64 bit precision
even after being transmitted through the network
'''
