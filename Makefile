# Scientific Machine Learning Benchmark
# A benchmark of regression models in chem- and materials informatics.
# Matthias Rupp 2019, Citrine Informatics.

# Run from main project directory

tests: force
	@echo 'Running tests...'
	pytest -rs tests

# install local developer version
develop: force
	@echo 'Building and installing local developer version...'
	python -m pip install -U -r requirements/all.txt 1> /dev/null
	python -m pip install -e . 1> /dev/null

# clean intermediate files
clean: force
	@echo 'Removing Python cache directories...'
	find . -name "__pycache__"  -type d -exec rm -r "{}" +
	find . -name ".ipynb_checkpoints" -type d -exec rm -r "{}" +
	find . -name ".pytest_cache" -type d -exec rm -r "{}" +

# reset to pristine state
purge: clean
	@echo 'Uninstalling developer version...'
	pip uninstall -y smlb 1> /dev/null
	@echo 'Removing build files...'
	find . -name ".eggs" -type d -exec rm -r "{}" +
	find . -name "build" -type d -exec rm -r "{}" +
	find . -name "dist" -type d -exec rm -r "{}" +
	find . -name "smlb.egg-info" -type d -exec rm -r "{}" +

force:
