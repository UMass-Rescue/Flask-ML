# Cybersecurity Institute Hackathon 2024

Welcome to the Second Annual Cybersecurity Institute Hackathon 2024. We will be using this repository to contain our RescueBox "tool-suite".

## Getting Started

### Step 1: Forking this repository

Begin by forking this repository to your GitHub account.

### Step 2: Create a directory for your open-source trust and safety tool

Under the directory `tool-suite`, create a directory for your project. For example, this could be `tool-suite/deep-fake-classification`.

### Step 3: Start Hacking!

See the [development](#development) section for more help on project setup.

### Step 4: Submit a PR back to this repository

When you're done, submit a PR back to this repository, so we can have everyone's tools ready to go!

## Development

### Create a virtual environment

```bash
python -m venv env
```

This uses `venv` to create a new virtual environment for you.

### Activate the virtual environment

For MacOS:

```bash
source env/bin/activate
```

For Windows:

```pwsh
.\env\Scripts\Activate.ps1
```

### Installing dependencies

Create a `requirements.txt` under your tool. For example, `tool-suite/audio-transcription/requirements.txt`

Then, run

```bash
pip install -e ".[dev]"
```

followed by

```bash
pip install -r tool-suite/audio-transcription/requirements.txt
```

### Running a sample task

Read the [README](./tool-suite/audio-transcription/README.md) in the tool-suite/audio-transcription directory for instructions on how to run the sample task.

### Flask-ML

This repository already contains a copy of the Python library [FlaskML](umass-rescue.github.io/Flask-ML/). Check out the website for more examples and usage documentation: https://umass-rescue.github.io/Flask-ML/
