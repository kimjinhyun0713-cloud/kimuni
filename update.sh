#!/bin/sh

rm -rf build dist *.egg-info
find . -type d -name "*.dist-info" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
pip install --upgrade setuptools wheel
python -m build
