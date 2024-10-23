---
sidebar_position: 2
---

# Writing a CLI

## Introduction

A CLI is helpful to run and debug code from the terminal. Flask-ML is able to automatically create a CLI for your machine learning code. Generating a CLI is a simple process that involves creating a single file.

## Prerequisites

- At least one endpoint in your server file must be decorated with `@server.route`, and the route must have a `task_schema_func` parameter. See [Adding a Task Schema](./getting-started#adding-a-ui-schema) for more information.

```python
@server.route("/transform_case", task_schema_func=create_transform_case_task_schema)
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    ...
```


## Server File

Ensure that your server file has a `if __name__ == "__main__":` block.

```python
# server.py
ml_server = MLServer(__name__)

if __name__ == "__main__":
    ml_server.run()
```

## CLI File

Create a new file called `cli.py` in your project directory.

```python
# cli.py

import argparse

from flask_ml.flask_ml_cli import MLCli
from server import ml_server
from small_blk_forensics.backend.server import server


def main():
    parser = argparse.ArgumentParser(description="Transform Case of multiple text inputs")
    cli = MLCli(server, parser)
    cli.run_cli()


if __name__ == "__main__":
    main()
```

That's it! Now, you can run your CLI with:

```bash
python cli.py
```
