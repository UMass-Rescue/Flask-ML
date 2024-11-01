---
sidebar_position: 4
---

# Adding Application Metadata

## Introduction

Adding application metadata such as the name, description, and version of your application is a simple process. Rescue-Box Desktop uses this metadata to display information about your application in the application list.

It is very simple to add metadata to your Flask-ML application. In your `server.py` file:

```python
from flask_ml.flask_ml import MLServer, load_file_as_string

server = MLServer(__name__)

server.add_app_metadata(
    name="Simple Server - Transform Case",
    author="Flask-ML Team",
    version="0.1.0",
    info=load_file_as_string("simple_server_info.md"),
)
```

Here, info is a Markdown string that contains an overview of your application. See [Sample Application Info](https://github.com/UMass-Rescue/Flask-ML/blob/master/simple_server_info.md) for an example.

This creates a new route at `/api/app_metadata` that returns the metadata as a JSON object. This will be used by [Rescue-Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop) to display it here:

![](/img/sample_markdown_app_metadata.png)
