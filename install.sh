#!/usr/bin/env bash

set -e

echo "📦 Updating system..."
sudo pacman -Syu --noconfirm

echo "📦 Installing required system packages (python, pip, gcc, etc)..."
sudo pacman -S --noconfirm python python-pip python-virtualenv gcc git

# (Optional) If you're using NVIDIA GPU and want full CodeCarbon GPU telemetry:
# sudo pacman -S --noconfirm nvidia-utils

echo "📁 Creating and activating Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "📄 Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing project Python dependencies..."
pip install fastapi uvicorn codecarbon SQLAlchemy pydantic requests python-dotenv databases

# If any package fails to install, try:
# pip install --no-cache-dir <pkg>

echo "✔ All dependencies installed!"

echo "🔧 Checking uvicorn installation..."
if ! command -v uvicorn &> /dev/null
then
    echo "❌ uvicorn NOT FOUND inside venv. Installation failed."
    exit 1
fi

echo "✔ uvicorn installed successfully."

echo "▶ Ready to run your FastAPI application!"
echo ""
echo "To start the server:"
echo ""
echo "    source venv/bin/activate"
echo "    uvicorn app.main:app --reload --port 8000"
echo ""
echo "Or run your run.py if you have one:"
echo ""
echo "    python run.py"
echo ""

echo "🎉 Setup complete!"
