#!/usr/bin/env bash

export PYTHONPATH="../blender_autocomplete/2.82"

pylint --extension-pkg-whitelist=cairo,cv2 modeling
