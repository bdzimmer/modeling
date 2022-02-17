#!/usr/bin/env bash

export PYTHONPATH="../blender_autocomplete/3.0"

pytest -s --tb=short --cov-report term-missing --cov-report html:htmlcov_tests --cov=modeling modeling

# xdg-open htmlcov_tests/index.html &
