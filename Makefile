install_project_lib:
	pip install -e .

install_requirements:
	@pipenv install -r requirements.txt

check_code:
	black project_lib/*.py scripts/*

mlflow:
	mlflow ui