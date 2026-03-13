.PHONY: setup check

setup:
	bash scripts/bootstrap_env.sh

check:
	python -m compileall src
