#!/bin/bash
# Activate virtual environment and set up Python path

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Setting Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Python version:"
python --version

echo "Python interpreter location:"
which python

echo "Virtual environment activated!"
echo "You can now run: python src/main.py"
echo "Or run the FastAPI server: python run.py"
echo ""
echo "For VS Code/Pyright:"
echo "1. Press Cmd+Shift+P"
echo "2. Type 'Python: Select Interpreter'"
echo "3. Choose './.venv/bin/python'"
echo "4. Restart VS Code if needed"
