# Flask-ML

## MLServer
The MLServer object is a wrapper class for the flask app object. It provides a decorator for turning a machine learning prediction function into a WebService on an applet.

## Installation

```
pip install flask-ml-server
```

### Usage
```Python3
from flask_ml_server.ml_server import MLServer
from encoder_decoder import DTypes
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# loading the Boston Housing Price Prediction dataset
data = load_boston()
# fitting a Linear Regression model to the housing data
model = LinearRegression()
model.fit(data.data, data.target)

# make a server instance
serv = MLServer(__name__)

# adding the "housing_price_prediction" endpoint to the server. 
# input_type and output_type are required parameters. They should be one among the available types in encoder_decoder.DTypes
@serv.route('/housing_price_prediction', input_type=DTypes.FLOAT_NDARRAY, output_type=DTypes.FLOAT_NDARRAY)
def predict_housing_price(img):
    '''
    img :: np.ndarray - array of features
    returns :: np.ndarray - housing price prediction
    '''
    return model.predict(img)

# begin server instance
serv.run()
```



## Authors
Jagath Jai Kumar
Prasanna Lakkur Subramanyam



This project is licensed under the MIT License - see the LICENSE file for details
