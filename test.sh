#!/usr/bin/env bash

export PYTHONPATH="../blender_autocomplete/3.0"

pytest -s --tb=short --cov-report term-missing --cov-report html:htmlcov/tests --cov=modeling modeling

xdg-open htmlcov/tests/index.html &
