# py -m venv .venv
# source .venv/Scripts/activate
# pip install -r requirements.txt
# clear
# flask run

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate virtual environment
source .venv/Scripts/activate

# Set Flask environment variables
export FLASK_APP=app.py            # Change to your actual Flask file if different
export FLASK_ENV=development       # Enables debug mode

# Start Flask server
flask run --host=0.0.0.0 --port=5000
