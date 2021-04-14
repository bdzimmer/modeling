#!/usr/bin/env bash

export PYTHONPATH="../blender_autocomplete/2.92"

pytest -s --tb=short --cov-report html:htmlcov_tests --cov=modeling

xdg-open htmlcov_tests/index.html &
