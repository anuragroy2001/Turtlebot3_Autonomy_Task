#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import json
import os
import re
import base64
import openai
import glob
from datetime import datetime
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import time as pytime
import threading
import torch
from ultralytics import YOLO

class SemanticPerceptionNode(Node):
    def __init__(self):
        super().__init__('semantic_perception')
        self.bridge = CvBridge()
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        if not os.getenv('OPENAI_API_KEY'):
            self.get_logger().error("OPENAI_API_KEY environment variable not set!")
            raise ValueError("OpenAI API key is required")
        
        # Thread safety
        self.processing_lock = threading.Lock()
        self.processing = False

        # Initialize YOLO model for precise object detection
        self.get_logger().info("Loading YOLO12 model...")
        try:
            self.yolo_model = YOLO("yolo12n.pt")  # Use YOLO12 nano version for speed
            self.get_logger().info("YOLO12 model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO11: {e}")
            self.get_logger().info("Continuing without YOLO - using OpenAI only mode")
            self.yolo_model = None

        # Camera initialization
        self.latest_depth = None
        self.camera_info = None
        self.latest_stamp = None
        self.latest_rgb = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Room/scene classification labels
        self.scene_labels = ["main office room", "break room", "bathroom"]
        
        # Object labels for filtering YOLO detections
        self.target_objects = ["desk", "chair", "bookshelf", "sofa", "refrigerator", "water dispenser", "toilet", "side table"]

        # Simplified semantic memory - only objects and scenes
        self.semantic_memory_file = self._get_next_memory_filename()
        self.semantic_memory = self.load_semantic_memory()
        
        # Processing control
        self.last_processing_time = 0
        self.processing_interval = 0.5  # Process every 0.5 seconds
        
        # Subscribers & Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/object_in_odom", 10)
        
        self.rgb_sub = self.create_subscription(
            Image, "/intel_realsense_r200_rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/intel_realsense_r200_depth/depth/image_raw", self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/intel_realsense_r200_depth/camera_info", self.camera_info_callback, 10)
        self.annotated_pub = self.create_publisher(Image, "/percep/annotated_image", 10)
        
        self.get_logger().info("Semantic Perception Node initialized with YOLO12 + OpenAI Vision API!")
    
    def encode_image_to_base64(self, cv_image):
        """Encode OpenCV image to base64 for OpenAI API"""
        _, buffer = cv2.imencode('.jpg', cv_image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def classify_with_openai(self, cv_image, yolo_bboxes):
        """Use OpenAI Vision API to label YOLO-detected objects and classify scene"""
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(cv_image)
            
            # Build prompt for YOLO bounding boxes (we only call this when YOLO has detections)
            bbox_info = "YOLO has detected the following bounding boxes (format: [x1, y1, x2, y2]):\n"
            for i, detection in enumerate(yolo_bboxes):
                bbox = detection['bbox']
                bbox_info += f"Box {i}: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] (confidence: {detection['confidence']:.2f})\n"
            
            self.get_logger().info(f"Sending {len(yolo_bboxes)} YOLO bounding boxes to OpenAI for labeling")
            
            prompt = f"""Analyze this image and provide comprehensive semantic understanding for office robot navigation.

{bbox_info}

ROOM TYPES (choose exactly one):
1. "main office room" - workspace with desks, chairs, computers, bookshelves
2. "break room" - relaxation area with sofas, refrigerator, water dispenser, coffee area
3. "bathroom" - restroom with toilet, sink, bathroom fixtures

INSTRUCTIONS:
1. SCENE TYPE: Classify this room as exactly one of: "main office room", "break room", or "bathroom"

2. YOLO OBJECT LABELING: For each YOLO bounding box listed above, identify what object is inside that box:
   - Look at the specific region defined by each bounding box coordinates
   - Label the main object within each bounding box
   - Use semantic labels like: desk, chair, sofa, refrigerator, water dispenser, bookshelf, toilet, sink, computer, etc.

3. SPATIAL LAYOUT: Describe the overall room layout and object arrangements

Please format your response as JSON:
{{
    "scene_type": "main office room|break room|bathroom",
    "yolo_objects": [
        {{"bbox_id": 0, "label": "object_name", "description": "what's in bounding box 0"}},
        {{"bbox_id": 1, "label": "object_name", "description": "what's in bounding box 1"}}
    ],
    "spatial_layout": "description of room layout and object arrangements",
    "confidence": 0.9
}}"""
            
            # Call OpenAI Vision API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using the latest vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Look for JSON content between curly braces
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    
                    # Validate required fields (flexible for both formats)
                    if 'scene_type' in result:
                        # Validate scene_type is one of our three labels
                        scene_type = result['scene_type']
                        if scene_type not in self.scene_labels:
                            # Map common variations to our labels
                            scene_mapping = {
                                'office': 'main office room',
                                'office room': 'main office room',
                                'workspace': 'main office room',
                                'break': 'break room',
                                'breakroom': 'break room',
                                'kitchen': 'break room',
                                'lounge': 'break room',
                                'restroom': 'bathroom',
                                'toilet': 'bathroom',
                                'washroom': 'bathroom'
                            }
                            scene_type = scene_mapping.get(scene_type.lower(), 'main office room')
                            result['scene_type'] = scene_type
                        
                        # Ensure objects field exists and convert yolo_objects to standard format
                        if 'objects' not in result:
                            result['objects'] = []
                        
                        # Convert yolo_objects to standard objects format
                        if 'yolo_objects' in result:
                            for yolo_obj in result['yolo_objects']:
                                if 'label' in yolo_obj:
                                    result['objects'].append({
                                        'name': yolo_obj['label'],
                                        'position': 'yolo-detected',
                                        'description': yolo_obj.get('description', ''),
                                        'bbox_id': yolo_obj.get('bbox_id', -1),
                                        'source': 'yolo_labeled'
                                    })
                        
                        self.get_logger().info(f"OpenAI classification successful: {result['scene_type']}")
                        return result
                
                # Fallback if JSON parsing fails
                self.get_logger().warn("Could not parse JSON from OpenAI response, using fallback")
                return {
                    "scene_type": "main office room",  # Default to main office room
                    "objects": [],
                    "spatial_layout": content[:200],
                    "confidence": 0.3
                }
                
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse OpenAI JSON response: {e}")
                return {
                    "scene_type": "main office room", 
                    "objects": [],
                    "spatial_layout": "parsing error",
                    "navigation_landmarks": [],
                    "semantic_labels": ["unknown"],
                    "confidence": 0.1
                }
                
        except Exception as e:
            self.get_logger().error(f"OpenAI Vision API error: {e}")
            return {
                "scene_type": "main office room",
                "objects": [],
                "spatial_layout": "error",
                "navigation_landmarks": [],
                "semantic_labels": ["unknown"],
                "confidence": 0.0
            }
    
    def _get_next_memory_filename(self):
        """Generate next numbered semantic memory filename in config directory"""
        # Create config directory if it doesn't exist
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            self.get_logger().info(f"Created config directory: {config_dir}")
        
        base_name = "semantic_exploration_memory"
        extension = ".json"
        
        # Find the highest existing number in config directory
        max_num = 0
        
        # Look for existing files with pattern semantic_exploration_memory_*.json in config dir
        pattern = os.path.join(config_dir, f"{base_name}_*.json")
        existing_files = glob.glob(pattern)
        
        for file in existing_files:
            try:
                # Extract number from filename like "config/semantic_exploration_memory_5.json"
                filename = os.path.basename(file)
                num_str = filename.replace(f"{base_name}_", "").replace(extension, "")
                if num_str.isdigit():
                    max_num = max(max_num, int(num_str))
            except:
                continue
        
        # Also check for the unnumbered file in config directory
        unnumbered_file = os.path.join(config_dir, f"{base_name}.json")
        if os.path.exists(unnumbered_file):
            max_num = max(max_num, 0)
        
        # Generate next filename with full path
        next_num = max_num + 1
        next_filename = os.path.join(config_dir, f"{base_name}_{next_num}.json")
        
        self.get_logger().info(f"Using semantic memory file: {next_filename}")
        return next_filename

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.img_frame = "realsense_depth_frame"
        # self.get_logger().info(f"Camera info received, using frame: {self.img_frame}")

    def load_semantic_memory(self):
        """Load existing semantic memory or create new simplified structure"""
        # Always create fresh memory for new numbered file
        self.get_logger().info(f"Creating fresh semantic memory for: {self.semantic_memory_file}")
        
        # Simplified semantic memory structure - objects and scenes only
        return {
            "objects": {},  # object_label: {positions: [], confidences: [], timestamps: [], best_position: {}}
            "scenes": {},   # scene_label: {unique_objects: [], object_count: int, confidences: [], timestamps: [], best_position: {computed_center}}
            "exploration_path": [],  # Track robot's exploration path
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "file_number": self.semantic_memory_file.split('_')[-1].replace('.json', ''),
                "node_version": "3.0_scene_center_computed",
                "structure": "objects_and_scenes_with_computed_centers"
            }
        }

    def save_semantic_memory(self):
        """Save current semantic memory to file"""
        try:
            with open(self.semantic_memory_file, 'w') as f:
                json.dump(self.semantic_memory, f, indent=4)
        except Exception as e:
            self.get_logger().error(f"Failed to save semantic memory: {e}")
# Add these debug lines to your callbacks:

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_stamp = msg.header.stamp
            # self.get_logger().info(f"RGB image received: {self.latest_rgb.shape}")  # Add this line
        except CvBridgeError as e:
            self.get_logger().error(f"RGB CvBridge error: {e}")
            return
        self._try_process_frame()

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            # self.get_logger().info(f"Depth image received: {self.latest_depth.shape}")  # Add this line
        except CvBridgeError as e:
            self.get_logger().error(f"Depth CvBridge error: {e}")
            return
        self._try_process_frame()

    def _try_process_frame(self):
        """Process frame with rate limiting and thread safety"""
        current_time = pytime.time()
        # self.get_logger().info(f"Try process frame called. RGB: {self.latest_rgb is not None}, Depth: {self.latest_depth is not None}")  # Add this line
        
        if (current_time - self.last_processing_time < self.processing_interval or 
            self.processing or self.latest_rgb is None or self.latest_depth is None):
            return
            
        with self.processing_lock:
            if not self.processing:
                self.processing = True
                self.last_processing_time = current_time
                # self.get_logger().info("Starting frame processing...")  # Add this line
                try:
                    self.process_frame()
                finally:
                    self.processing = False

    def process_frame(self):
        """Hybrid processing: YOLO for bounding boxes + OpenAI for scene understanding and labeling"""
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            self.get_logger().warn("Missing data for processing")
            return

        # Simple depth threshold check - only process if objects are within 2.5m
        center_depth = self.latest_depth[self.latest_depth.shape[0]//2, self.latest_depth.shape[1]//2]
        if np.isnan(center_depth) or center_depth > 3.0:
            self.get_logger().info(f"Scene too far ({center_depth:.2f}m), skipping processing")
            return

        cv_img = self.latest_rgb.copy()
        self.get_logger().info(f"Processing image with YOLO11 + OpenAI: {cv_img.shape}, depth: {center_depth:.2f}m")
        
        # Step 1: YOLO object detection for bounding boxes only (no labels used)
        yolo_detections = []
        if self.yolo_model is not None:
            yolo_detections = self.detect_objects_with_yolo(cv_img)
            self.get_logger().info(f"YOLO detected {len(yolo_detections)} bounding boxes for OpenAI labeling")
            
            # If YOLO doesn't detect anything, skip processing entirely
            if len(yolo_detections) == 0:
                self.get_logger().info("No YOLO detections found → skipping all processing")
                return
        else:
            self.get_logger().info("YOLO model not available → skipping processing")
            return

        # Step 2: OpenAI scene classification and object labeling (only for YOLO boxes)
        openai_result = self.classify_with_openai(cv_img, yolo_detections)
        scene_label = openai_result.get('scene_type', 'unknown')
        scene_confidence = openai_result.get('confidence', 0.0)
        
        if scene_confidence < 0.3:
            self.get_logger().info("Low confidence scene analysis")
            self._publish_annotated_image(cv_img, [], [])
            return
            
        self.get_logger().info(f"OpenAI Scene: {scene_label} (confidence: {scene_confidence:.3f})")
        
        # Step 3: Process YOLO-detected objects with OpenAI labels
        detected_objects = []
        valid_boxes = []
        
        for obj in openai_result.get('objects', []):
            object_name = obj.get('name', 'unknown')
            
            # All objects should have bbox_id since we only process YOLO detections
            if 'bbox_id' in obj and obj['bbox_id'] >= 0:
                bbox_id = obj['bbox_id']
                if bbox_id < len(yolo_detections):
                    bbox = yolo_detections[bbox_id]['bbox']
                    yolo_confidence = yolo_detections[bbox_id]['confidence']
                    
                    # Get precise 3D position using YOLO bounding box
                    position_3d = self._get_3d_position(bbox, self.latest_depth)
                    
                    if position_3d:
                        detection_data = {
                            'type': 'object',
                            'label': object_name,
                            'confidence': min(scene_confidence, yolo_confidence),
                            'position': position_3d,
                            'timestamp': datetime.now().isoformat(),
                            'bbox': bbox,
                            'scene_type': scene_label,
                            'detection_method': 'yolo_bbox_openai_labeled'
                        }
                        detected_objects.append(detection_data)
                        valid_boxes.append(bbox)
                        self._update_object_memory(object_name, detection_data)
                        
                        self.get_logger().info(f"YOLO+OpenAI object {object_name}: ({position_3d['x']:.2f}, {position_3d['y']:.2f}, {position_3d['z']:.2f})")
                    else:
                        self.get_logger().warn(f"Could not get 3D position for {object_name} in bbox {bbox_id}")
                else:
                    self.get_logger().warn(f"Invalid bbox_id {bbox_id} for object {object_name}")
            else:
                self.get_logger().warn(f"Object {object_name} missing bbox_id - skipping")

        # Step 4: Compute scene center from unique objects and update semantic memory
        if scene_confidence > 0.3:
            scene_center = self._calculate_scene_center(detected_objects)
            if scene_center:
                # Get unique object labels only (no duplicates for scene association)
                unique_objects = {}
                for obj in detected_objects:
                    label = obj['label']
                    if label not in unique_objects or obj['confidence'] > unique_objects[label]['confidence']:
                        unique_objects[label] = obj
                
                unique_object_labels = list(unique_objects.keys())  # Only unique labels
                
                scene_data = {
                    'type': 'scene',
                    'label': scene_label,
                    'confidence': scene_confidence,
                    'position': scene_center,  # This is the computed center from unique objects
                    'timestamp': datetime.now().isoformat(),
                    'objects_detected': len(unique_object_labels),  # Count unique objects
                    'objects': unique_object_labels,  # Only unique object labels
                    'detection_method': 'computed_center_from_unique_objects'
                }
                self._update_scene_memory(scene_label, scene_data)
                
                self.get_logger().info(f"✓ Scene '{scene_label}' center computed: ({scene_center['x']:.3f}, {scene_center['y']:.3f}, {scene_center['z']:.3f})")
                self.get_logger().info(f"  Unique objects used: {unique_object_labels}")
            else:
                self.get_logger().warn(f"Could not compute scene center for '{scene_label}' - no valid object positions")

        # Step 5: Save semantic memory and publish visualization
        self.save_semantic_memory()
        self._publish_annotated_image(cv_img, detected_objects, valid_boxes)
        
        # Log comprehensive understanding of the hybrid workflow
        yolo_objs = [obj['label'] for obj in detected_objects if obj.get('detection_method') == 'yolo_bbox_openai_labeled']
        self.get_logger().info(f"HYBRID RESULT: {scene_label} | YOLO boxes→OpenAI labels: {yolo_objs}")
        
        if yolo_objs:
            self.get_logger().info(f"✓ YOLO detected {len(valid_boxes)} bounding boxes → OpenAI labeled as: {yolo_objs}")
        else:
            self.get_logger().info("✓ No valid objects detected from YOLO bounding boxes")

        # Clear processed frames
        self.latest_rgb = None
        self.latest_depth = None

    def _estimate_object_position(self, position_desc, base_depth):
        """Estimate 3D position from textual description"""
        try:
            # Parse position description like "left-far", "center-near", etc.
            parts = position_desc.lower().split('-')
            
            # Horizontal offset
            x_offset = 0.0
            if 'left' in parts:
                x_offset = -1.0
            elif 'right' in parts:
                x_offset = 1.0
            
            # Depth offset  
            z_offset = base_depth
            if 'far' in parts:
                z_offset = base_depth + 1.0
            elif 'near' in parts:
                z_offset = max(0.5, base_depth - 0.5)
            
            # Transform to odom coordinates (simplified)
            return {
                "x": float(z_offset),  # Forward is X in robot frame
                "y": float(x_offset),  # Left/right is Y
                "z": 0.0               # Ground level
            }
            
        except Exception as e:
            self.get_logger().error(f"Position estimation error: {e}")
            return {"x": 2.0, "y": 0.0, "z": 0.0}  # Default forward position

    def _get_3d_position(self, box, depth_image):
        """Convert 2D bounding box to 3D position"""
        try:
            x_min, y_min, x_max, y_max = map(int, box)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            
            # Sample multiple depth points for robustness
            depth_samples = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y_sample = max(0, min(y_center + dy, depth_image.shape[0] - 1))
                    x_sample = max(0, min(x_center + dx, depth_image.shape[1] - 1))
                    depth_val = depth_image[y_sample, x_sample]
                    if not np.isnan(depth_val) and depth_val > 0.0:
                        depth_samples.append(depth_val)
            
            if not depth_samples:
                return None
                
            depth = np.median(depth_samples)
            
            # Convert to 3D coordinates
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            X = (x_center - cx) * depth / fx
            Y = (y_center - cy) * depth / fy
            Z = depth

            # Transform to odom frame - USE LATEST TRANSFORM TIME
            pose = PoseStamped()
            pose.header.stamp = rclpy.time.Time().to_msg()  # Use current time instead
            pose.header.frame_id = self.img_frame
            pose.pose.position.x = float(X)
            pose.pose.position.y = float(Y)
            pose.pose.position.z = float(Z)
            pose.pose.orientation.w = 1.0

            try:
                # Use latest available transform
                odom_pose = self.tf_buffer.transform(pose, "odom", timeout=Duration(seconds=1.0))
                return {
                    "x": odom_pose.pose.position.x,
                    "y": odom_pose.pose.position.y,
                    "z": odom_pose.pose.position.z
                }
            except Exception as e:
                self.get_logger().error(f"TF transform failed: {e}")
                # Return camera-frame coordinates as fallback
                return {
                    "x": float(X),
                    "y": float(Y), 
                    "z": float(Z)
                }
                
        except Exception as e:
            self.get_logger().error(f"3D position calculation error: {e}")
            return None

    def _calculate_scene_center(self, detected_objects):
        """Calculate center position from unique objects only (prevents duplicate bias)"""
        if not detected_objects:
            self.get_logger().warn("No detected objects for scene center calculation")
            return None
        
        # Group objects by label and keep only the highest confidence detection per object type
        unique_objects = {}
        for obj in detected_objects:
            label = obj['label']
            if label not in unique_objects or obj['confidence'] > unique_objects[label]['confidence']:
                unique_objects[label] = obj
        
        # Extract positions from unique objects only
        unique_positions = []
        object_details = []
        
        for label, obj in unique_objects.items():
            if obj.get('position'):
                pos = obj['position']
                unique_positions.append(pos)
                object_details.append(f"{label}({obj['confidence']:.2f})")
        
        if not unique_positions:
            self.get_logger().warn("No valid positions from unique objects")
            return None
            
        # Calculate centroid from unique object positions
        center_x = sum(pos['x'] for pos in unique_positions) / len(unique_positions)
        center_y = sum(pos['y'] for pos in unique_positions) / len(unique_positions)
        center_z = sum(pos['z'] for pos in unique_positions) / len(unique_positions)
        
        computed_center = {"x": center_x, "y": center_y, "z": center_z}
        
        self.get_logger().info(f"Scene center computed from {len(unique_positions)} unique objects: {object_details}")
        self.get_logger().info(f"Computed center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
        
        return computed_center

    def _update_object_memory(self, label, detection_data):
        """Update object detection memory"""
        if label not in self.semantic_memory["objects"]:
            self.semantic_memory["objects"][label] = {
                "positions": [],
                "confidences": [],
                "timestamps": [],
                "highest_confidence": 0.0,
                "best_position": None
            }
        
        memory = self.semantic_memory["objects"][label]
        memory["positions"].append(detection_data["position"])
        memory["confidences"].append(detection_data["confidence"])
        memory["timestamps"].append(detection_data["timestamp"])
        
        # Update best detection
        if detection_data["confidence"] > memory["highest_confidence"]:
            memory["highest_confidence"] = detection_data["confidence"]
            memory["best_position"] = detection_data["position"]
        
        # Keep only recent detections (last 100)
        if len(memory["positions"]) > 100:
            memory["positions"] = memory["positions"][-100:]
            memory["confidences"] = memory["confidences"][-100:]
            memory["timestamps"] = memory["timestamps"][-100:]

    def _update_scene_memory(self, label, scene_data):
        """Update scene classification memory with computed center from unique objects"""
        if label not in self.semantic_memory["scenes"]:
            self.semantic_memory["scenes"][label] = {
                "unique_objects": [],
                "object_count": 0,
                "confidences": [],
                "timestamps": [],
                "highest_confidence": 0.0,
                "best_position": None  # This will be the computed center from unique objects
            }
        
        memory = self.semantic_memory["scenes"][label]
        
        # Store unique object list and computed center position
        memory["unique_objects"] = scene_data["objects"]  # Only unique object labels
        memory["object_count"] = scene_data["objects_detected"]
        memory["confidences"].append(scene_data["confidence"])
        memory["timestamps"].append(scene_data["timestamp"])
        
        # Update best detection and use computed center as the scene position
        if scene_data["confidence"] > memory["highest_confidence"]:
            memory["highest_confidence"] = scene_data["confidence"]
            memory["best_position"] = scene_data["position"]  # This is the computed center
            
        # Keep only recent detections (last 50 for scenes)
        if len(memory["confidences"]) > 50:
            memory["confidences"] = memory["confidences"][-50:]
            memory["timestamps"] = memory["timestamps"][-50:]

    def detect_objects_with_yolo(self, cv_image):
        """Use YOLO for bounding box detection only - NO LABELS, only coordinates"""
        try:
            results = self.yolo_model(cv_image, conf=0.25, verbose=False)
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores = boxes.conf.cpu().numpy()
                
                # Return only bounding boxes with confidence - NO YOLO LABELS USED
                for i, (box, score) in enumerate(zip(xyxy, scores)):
                    if score > 0.25:  # Confidence threshold
                        detections.append({
                            'bbox': box.tolist(),  # [x1, y1, x2, y2] - coordinates only
                            'confidence': float(score),  # YOLO detection confidence
                            'detection_id': i  # Just for tracking, no semantic meaning
                        })
                
                self.get_logger().info(f"YOLO detected {len(detections)} objects → sending bounding boxes to OpenAI for labeling")
            else:
                self.get_logger().info("No objects detected by YOLO → OpenAI will analyze entire scene")
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f"YOLO detection error: {e}")
            return []

    def _publish_detections(self, detected_objects, scene_label):
        """Publish detection results"""
        if detected_objects:
            # Publish best object detection
            best_object = max(detected_objects, key=lambda x: x['confidence'])
            
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "odom"
            pose.pose.position.x = best_object['position']['x']
            pose.pose.position.y = best_object['position']['y']
            pose.pose.position.z = best_object['position']['z']
            pose.pose.orientation.w = 1.0
            
            self.pose_pub.publish(pose)

    def _publish_annotated_image(self, image, detected_objects, valid_boxes):
        """Enhanced visualization for YOLO + OpenAI hybrid workflow"""
        annotated_img = image.copy()
        
        # Draw YOLO bounding boxes
        for i, box in enumerate(valid_boxes):
            if isinstance(box, list) and len(box) == 4:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Add small label on bounding box
                cv2.putText(annotated_img, f"YOLO", (x_min, y_min-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Create text labels from detected objects
        labels = []
        
        # Add scene information
        scene_info = f"SCENE: {detected_objects[0].get('scene_type', 'unknown') if detected_objects else 'none'}"
        labels.append(scene_info)
        
        # Add object detection summary
        for i, obj in enumerate(detected_objects[:5]):  # Limit to first 5 objects
            method = "🎯YOLO+OpenAI" if obj.get('detection_method') == 'yolo_bbox_openai_labeled' else "🤖OpenAI"
            label_text = f"{method} {obj['label']}: {obj['confidence']:.2f}"
            labels.append(label_text)
        
        # Draw text labels on image
        for i, label in enumerate(labels):
            y_position = 30 + (i * 25)
            color = (255, 0, 0) if label.startswith("SCENE:") else (0, 255, 0)
            if "🎯YOLO+OpenAI" in label:
                color = (0, 255, 255)  # Yellow for YOLO+OpenAI hybrid detections
            elif "🤖OpenAI" in label:
                color = (255, 0, 255)  # Magenta for OpenAI-only estimations
            
            cv2.putText(annotated_img, label, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        try:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8"))
        except CvBridgeError as e:
            self.get_logger().error(f"Annotated image publish error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down semantic perception node...")
        node.save_semantic_memory()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()