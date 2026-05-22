#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "Checking for NEDAS Framework & Tutorial Updates..."
echo "=================================================="

# 1. Update NEDAS Core framework
if [ -d "/home/appuser/NEDAS/.git" ]; then
    echo "Updating NEDAS core framework..."
    cd /home/appuser/NEDAS
    # Fetch and merge updates without breaking local editable pip structures
    git pull
else
    echo "Warning: /home/appuser/NEDAS/.git not found. Skipping framework update."
fi

# 2. Manage and Update NEDAS Tutorials
TUTORIALS_DIR="/home/appuser/work/NEDAS_tutorials"

if [ -d "$TUTORIALS_DIR/.git" ]; then
    echo "Updating existing NEDAS tutorials repository..."
    cd "$TUTORIALS_DIR"
    git pull
else
    echo "Tutorials repository missing or detached. Re-cloning..."
    # Safe backup/creation of the directory tree
    mkdir -p "/home/appuser/work"
    git clone https://github.com/myying/NEDAS_tutorials.git "$TUTORIALS_DIR"
fi

# 3. Navigate into the work target directory
cd "$TUTORIALS_DIR"

echo "=================================================="
echo "Starting Jupyter Lab Environment..."
echo "=================================================="

# exec overrides the shell process so Jupyter receives shutdown signals correctly
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --ServerApp.token="${JUPYTER_TOKEN:-}" \
    --ServerApp.password=''
