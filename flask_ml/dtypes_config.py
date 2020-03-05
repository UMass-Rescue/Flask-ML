# from dtypes_encode_decode import required encoders/decoders
# from dtypes_extract_wrap import required extract/wrap functions


decoders = {
    InputTypes.IMAGE: img_decode,
    InputTypes.IMAGE_BATCH: img_batch_decode,
    InputTypes.STRING: string_decode, # can be an identity function if decoding not required. 
                                      # This is so that all decoding functions follow the same pattern.
                                      # The same code can be reused.
    InputTypes.FLOAT_NDARRAY: float_ndarray_decode,
    InputTypes.FLOAT: float_decode # identify function if decoding not required
}

encoders = {
    OutputTypes.IMAGE: img_encode,
    OutputTypes.STRING: string_encode, # can be an identity function if encoding not required. 
                                      # This is so that all encoding functions follow the same pattern.
                                      # The same code can be reused.
    OutputTypes.FLOAT_NDARRAY: float_ndarray_encode,
    OutputTypes.FLOAT: float_encode # identify function if encoding not required
}


# extractors take Request object and return the input to the decoder
extract_input = {
    InputTypes.IMAGE: img_extract,
    InputTypes.IMAGE_BATCH: img_batch_extract,
    InputTypes.STRING: string_extract,
    InputTypes.FLOAT_NDARRAY: float_ndarray_extract,
    InputTypes.FLOAT: float_extract # identify function if encoding not required
}

# wrap output takes the encoded output and places it in the appropritate position in the response
wrap_output = {
    OutputTypes.IMAGE: img_wrap,
    OutputTypes.STRING: string_wrap, # can be an identity function if encoding not required. 
                                      # This is so that all encoding functions follow the same pattern.
                                      # The same code can be reused.
    OutputTypes.FLOAT_NDARRAY: float_ndarray_wrap,
    OutputTypes.FLOAT: float_wrap # identify function if encoding not required
}


'''
Procedure to add support for a new data type - 

1. Add data type in InputTypes and/or OutputTypes enum
2. Add encoding and decoding functions in dtypes_encode_decode.py
3. Add extract input and wrap output functions in dtypes_extract_wrap.py
4. Configure mapping from dtype to encode/decode function in dtypes_config.py .
4. Configure extract_input and wrap_output in dtypes_config.py .
'''