.PHONY: install lint format test train run-api run-ui docker-up clean

install:
	.\.venv\Scripts\pip.exe install -r requirements.txt

lint:
	.\.venv\Scripts\ruff.exe check src/ api/ tests/

format:
	.\.venv\Scripts\black.exe src/ api/ tests/ dashboard/

test:
	.\.venv\Scripts\pytest.exe tests/ -v --tb=short

train:
	.\.venv\Scripts\python.exe -m src.model_training

run-api:
	.\.venv\Scripts\uvicorn.exe api.app:app --reload --port 8000

run-ui:
	.\.venv\Scripts\streamlit.exe run dashboard/streamlit_app.py

docker-up:
	docker-compose up --build

clean:
	rm -rf __pycache__ .pytest_cache mlruns
