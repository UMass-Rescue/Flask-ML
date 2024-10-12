SRC = $(wildcard *.py) $(shell find src tests -type f -name '*.py')

format:
	black --line-length 110 $(SRC)
	isort --profile black $(SRC)
	flake8 --max-line-length=110 --ignore=E203,W503 --select=F,N $(SRC)
	mypy $(SRC)

generate-models:
	datamodel-codegen --input src/flask_ml/flask_ml_server/openapi.yaml --input-file-type openapi --output src/flask_ml/flask_ml_server/_generated_models.py --output-model-type pydantic_v2.BaseModel --field-constraints --snake-case-field --keep-model-order --use-annotated --capitalize-enum-members
