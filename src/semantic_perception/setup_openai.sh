#!/bin/bash

# Setup script for semantic perception with OpenAI Vision API
echo "Setting up semantic perception node with OpenAI Vision API..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY environment variable is not set!"
    echo "Please set it using: export OPENAI_API_KEY='your-api-key-here'"
    echo "Or add it to your ~/.bashrc file"
else
    echo "OPENAI_API_KEY is set ✓"
fi

# Download YOLO model if needed
echo "Downloading YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolo12n.pt')"

echo "Setup complete!"
echo ""
echo "To run the semantic perception node:"
echo "1. Make sure OPENAI_API_KEY is set in your environment"
echo "2. Source your ROS2 workspace: source install/setup.bash"
echo "3. Run: ros2 run vlm_perception_pkg semantic_perception_new"
