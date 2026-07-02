#!/bin/bash
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "venv not found or broken. Creating new venv..."
    rm -rf venv
    python3.12 -m venv venv || python3 -m venv venv || { echo "Error: Python 3.10+ not found. Install: brew install python@3.12"; exit 1; }
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
fi

exec venv/bin/python app.py
