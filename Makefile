.PHONY: setup analysis test all clean

setup:
	python -m pip install -r requirements.txt

analysis:
	python analysis/run_all.py

test:
	python -m pytest -q

all: analysis test

clean:
	rm -rf results/figures results/tables results/summary.json
