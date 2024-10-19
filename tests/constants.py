from flask_ml.flask_ml_server.models import *

TEXT_INPUT_SCHEMA = InputSchema(key="text_input", label="Text Input", input_type=InputType.TEXT)
TEXTAREA_INPUT_SCHEMA = InputSchema(key="text_input", label="Text Area Input", input_type=InputType.TEXTAREA)
FILE_INPUT_SCHEMA = InputSchema(key="file_input", label="File Input", input_type=InputType.FILE)
BATCHTEXT_INPUT_SCHEMA = InputSchema(
    key="text_inputs", label="Batch Text Inputs", input_type=InputType.BATCHTEXT
)
BATCHFILE_INPUT_SCHEMA = InputSchema(
    key="file_inputs", label="Batch File Inputs", input_type=InputType.BATCHFILE
)
DIRECTORY_INPUT_SCHEMA = InputSchema(key="dir_input", label="Directory Input", input_type=InputType.DIRECTORY)
BATCHDIRECTORY_INPUT_SCHEMA = InputSchema(
    key="dir_inputs", label="Batch Directory Inputs", input_type=InputType.BATCHDIRECTORY
)

TEXT_PARAM_SCHEMA = ParameterSchema(
    key="param1",
    label="Text Parameter",
    value=TextParameterDescriptor(parameter_type=ParameterType.TEXT, default="default"),
)
ENUM_PARAM_SCHEMA = ParameterSchema(
    key="param1",
    label="Enum Parameter",
    value=EnumParameterDescriptor(
        parameter_type=ParameterType.ENUM,
        enum_vals=[EnumVal(label="Option 1", key="option_1"), EnumVal(label="Option 2", key="option_2")],
        default="option_1",
    ),
)
FLOAT_PARAM_SCHEMA = ParameterSchema(
    key="param1",
    label="Float Parameter",
    value=FloatParameterDescriptor(parameter_type=ParameterType.FLOAT, default=0.0),
)
INT_PARAM_SCHEMA = ParameterSchema(
    key="param1",
    label="Int Parameter",
    value=IntParameterDescriptor(parameter_type=ParameterType.INT, default=1),
)
RANGED_FLOAT_PARAM_SCHEMA = ParameterSchema(
    key="param1",
    label="Ranged Float Parameter",
    value=RangedFloatParameterDescriptor(
        parameter_type=ParameterType.RANGED_FLOAT,
        default=0.0,
        range=FloatRangeDescriptor(min=0.0, max=1.0),
    ),
)
RANGED_INT_PARAM_SCHEMA = ParameterSchema(
    key="param1",
    label="Ranged Int Parameter",
    value=RangedIntParameterDescriptor(
        parameter_type=ParameterType.RANGED_INT,
        default=0,
        range=IntRangeDescriptor(min=0, max=10),
    ),
)
