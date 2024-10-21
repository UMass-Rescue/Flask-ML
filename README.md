# Flask-ML

![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
[![codecov](https://codecov.io/github/UMass-Rescue/Flask-ML/graph/badge.svg?token=DOXIBULQQS)](https://codecov.io/github/UMass-Rescue/Flask-ML)

Flask-ML helps users easily deploy their ML models as a web service. Flask-ML Server, similar to Flask, allows the user to specify web services using the decorator pattern. But Flask-ML allows users to specify the input schema being expected by the ML function and provides helper classes to form response objects for the outputs produced by the ML function. Furthermore, users can specify schemas for their inputs and outputs in order to expose their model via the frontend provided by [RescueBox-Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop).

### Installation

To install Flask-ML
```
pip install flask-ml
```

### Usage examples

#### Server

Refer server_example.py

#### Client

Refer client_example.py

#### Development

Install both production and dev dependencies
```
pip install -e ".[dev]"
```

To re-generate the model classes, run
```
make generate-models
```
