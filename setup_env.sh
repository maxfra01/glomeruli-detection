#!/bin/bash

ENV_NAME=".venv"

echo "Creating virtual environment in $ENV_NAME"
python3 -m venv $ENV_NAME

echo "Activating virtual environment"
source $ENV_NAME/bin/activate

echo "Installing dependencies from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete. Environment is active."
