#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Activating venv ==="
source venv/bin/activate

echo "=== Cleaning old build ==="
rm -rf build dist

echo "=== Building MacDictator.app ==="
python setup.py py2app

echo ""
echo "=== Done! ==="
echo "App: dist/MacDictator.app"
echo ""
echo "To run:  open dist/MacDictator.app"
