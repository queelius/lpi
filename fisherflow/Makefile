.PHONY: venv install test clean

venv:
	python3 -m venv venv
	@echo "Run 'source venv/bin/activate' to activate"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info venv