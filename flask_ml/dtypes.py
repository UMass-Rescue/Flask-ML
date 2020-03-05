from enum import Enum, auto

class InputTypes(Enum):
    IMAGE = auto() # 3d numpy array
    IMAGE_BATCH = auto() # 4d numpy array
    STRING = auto() # python string
    FLOAT_NDARRAY = auto() # numpy ndarray
    FLOAT = auto() # python float or numpy float. How about anything that converts to float using "float" method? TODO

class OutputTypes(Enum):
    STRING = auto() # python string
    IMAGE = auto() # 3d numpy array
    FLOAT_NDARRAY = auto() # numpy ndarray
    FLOAT = auto() # python float or numpy float. How about anything that converts to float using "float" method? TODO