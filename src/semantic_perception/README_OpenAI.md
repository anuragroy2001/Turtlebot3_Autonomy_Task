# Semantic Perception with OpenAI Vision API

This ROS2 node provides enhanced semantic understanding for robot exploration using YOLO12 object detection combined with OpenAI's Vision API for rich semantic classification.

## Features

- **YOLO12 Object Detection**: Fast and accurate object detection
- **OpenAI Vision API**: Rich semantic understanding of scenes and objects
- **3D Position Mapping**: Converts 2D detections to 3D world coordinates
- **Semantic Memory**: Session-based storage of semantic exploration data
- **Spatial Clustering**: Groups similar detections to reduce duplicates
- **Enhanced Scene Understanding**: Contextual scene classification with spatial relationships

## Prerequisites

1. **OpenAI API Key**: You need a valid OpenAI API key
2. **ROS2**: Humble or later
3. **Python Dependencies**: Listed in `requirements.txt`

## Setup

1. **Install dependencies**:
   ```bash
   cd /path/to/vlm_perception_pkg
   ./setup_openai.sh
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or add to your `~/.bashrc`:
   ```bash
   echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Build the ROS2 package**:
   ```bash
   cd /path/to/your/workspace
   colcon build --packages-select vlm_perception_pkg
   source install/setup.bash
   ```

## Usage

### Running the Node

```bash
ros2 run vlm_perception_pkg semantic_perception_new
```

### Topics

**Subscribed Topics:**
- `/intel_realsense_r200_rgb/image_raw` - RGB camera feed
- `/intel_realsense_r200_depth/depth/image_raw` - Depth camera feed  
- `/intel_realsense_r200_depth/camera_info` - Camera intrinsics

**Published Topics:**
- `/object_in_odom` - Best detected object position in odom frame
- `/semantic_location` - Semantic location clusters
- `/percep/annotated_image` - Annotated image with detections

### Semantic Memory

The node creates session-based semantic memory files:
- `semantic_exploration_memory_1.json`
- `semantic_exploration_memory_2.json`
- etc.

Each file contains:
- Object detections with 3D positions
- Scene classifications
- Semantic location clusters
- Exploration metadata

### Configuration

The node provides several parameters for tuning:
- `confidence_threshold`: Minimum confidence for detections (default: 0.5)
- `cluster_eps`: Spatial clustering distance threshold (default: 1.0m)
- `cluster_min_samples`: Minimum samples for clustering (default: 2)
- `detection_cooldown`: Time between detections (default: 5.0s)

## OpenAI Integration Details

### Scene Understanding
The OpenAI Vision API provides:
- Scene type classification (office, kitchen, bathroom, etc.)
- Object identification with semantic context
- Spatial relationship understanding
- Rich semantic labels for navigation

### Enhanced Object Detection
- YOLO12 provides fast, accurate bounding boxes
- OpenAI enhances with semantic meaning and context
- Combined confidence scoring
- Spatial clustering reduces duplicate detections

### Semantic Memory Structure
```json
{
  "objects": {
    "desk": {
      "positions": [...],
      "confidences": [...],
      "highest_confidence": 0.95,
      "best_position": {"x": 1.2, "y": 0.5, "z": 0.8}
    }
  },
  "scenes": {
    "office": {
      "positions": [...],
      "semantic_labels": ["workspace", "computer", "desk"],
      "spatial_context": "..."
    }
  },
  "locations": {
    "loc_0": {
      "scene_type": "office",
      "objects": ["desk", "chair", "computer"],
      "center": {"x": 1.0, "y": 0.0, "z": 0.5},
      "confidence": 0.85
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'openai'**
   - Run: `pip install openai>=1.0.0`

2. **OpenAI API Key Error**
   - Ensure OPENAI_API_KEY environment variable is set
   - Check API key validity

3. **YOLO Model Download Issues**
   - Run: `python3 -c "from ultralytics import YOLO; YOLO('yolo12n.pt')"`
   - Ensure internet connectivity

4. **TF Transform Errors**
   - Ensure camera and robot TF frames are properly configured
   - Check that tf2 is running

### Performance Tuning

- **OpenAI API Costs**: Reduce image detail level or processing frequency
- **Processing Speed**: Adjust `processing_interval` for faster/slower processing
- **Memory Usage**: Limit detection history in semantic memory

## Cost Considerations

OpenAI Vision API charges per image processed. To minimize costs:
- Increase processing intervals
- Use lower detail images when possible
- Implement smart triggering (only process on significant scene changes)
- Consider caching results for similar scenes

## Examples

See the semantic memory JSON files generated during operation for examples of the rich semantic understanding provided by this system.
