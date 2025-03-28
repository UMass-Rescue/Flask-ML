# Server deployment instructions

Let's consider the following example of a housing price prediction model being deployed using Flask-ML. The following is the server code listing - 

```Python3
# filename - server.py
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

## To start the  *development server*, we have the following options - 

1. As the above code listing contains "serv.run()", we can just run the python file in order to start the server.
```
python server.py
```

2. Start the server using "flask run" command. In order to use the "flask run" command (in its default configuration), we need to have a variable called "app" or "application" which points to a Flask application instance. MLServer contains a variable called "app" which is exactly what we need. At the end of the code in the above listing, make the following changes

```
# begin server instance
# serv.run()     # commented out as this is no longer needed
app = serv.app
```

After this, we need to export the filename of our python file containing this "app" variable (in our case, the filename is "server.py"). Run the following on the command line -

```
export FLASK_APP=server.py
flask run
```
The development server should now be running.

## Production server deployment

The default **development** server that comes with Flask isn't suitable for production deployment. Hence, we need to use a different production grade server for production deployments. 

In our example, we will use [gunicorn](https://gunicorn.org/). To install gunicorn, run - 

```
pip install gunicorn
```

Even with gunicorn, we need to specify which variable in our file contains the Flask application instance. Firstly, we need to make the same changes to the code as we did for "flask run" command - 

```
# begin server instance
# serv.run()     # commented out as this is no longer needed
app = serv.app
```

We then run the following command to start the gunicorn server - 

```
gunicorn -b localhost:8000 -w 4 server:app
```

"The -b option tells gunicorn where to listen for requests, which I set to the internal network interface at port 8000. It is usually a good idea to run Python web applications without external access, and then have a very fast web server that is optimized to serve static files accepting all requests from clients. This fast web server will serve static files directly, and forward any requests intended for the application to the internal server. I will show you how to set up nginx as the public facing server in the next section.

The -w option configures how many workers gunicorn will run. Having four workers allows the application to handle up to four clients concurrently, which for a web application is usually enough to handle a decent amount of clients, since not all of them are constantly requesting content. Depending on the amount of RAM your server has, you may need to adjust the number of workers so that you don't run out of memory.

The server:app argument tells gunicorn how to load the application instance. The name before the colon is the module that contains the application, and the name after the colon is the name of this application (this is the variable in your python file pointing to the Flask application instance)."

(Majority of the above three paragraphs comes from - https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-xvii-deployment-on-linux  )

