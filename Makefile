.PHONY: install train api ui test docker

install:
	pip install -e ".[ui,dev]"

train:
	python -m explicita.train

api:
	uvicorn explicita.api:app --reload

ui:
	streamlit run ui/app.py

test:
	pytest -q

docker:
	docker build -t explicita .
	docker run --rm -p 8000:8000 explicita
