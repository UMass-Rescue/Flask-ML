SRC = $(wildcard *.py) $(shell find src tests -type f -name '*.py')

all: test

format:
	black --line-length 110 $(SRC)
	isort --profile black $(SRC)
	flake8 --max-line-length=110 --ignore=E203,W503 --select=F,N $(SRC)
	# mypy $(SRC)

autoflake8:
	autoflake8 --in-place $(SRC)

test:
	pytest tests/* --cov=flask_ml --cov-report=xml --cov-report term --cov-report=html

generate-models:
	datamodel-codegen --input src/flask_ml/flask_ml_server/openapi.yaml --input-file-type openapi --output src/flask_ml/flask_ml_server/models.py --output-model-type pydantic_v2.BaseModel --field-constraints --snake-case-field --use-annotated --capitalize-enum-members --set-default-enum-member --use-default --enum-field-as-literal one --allow-population-by-field-name
