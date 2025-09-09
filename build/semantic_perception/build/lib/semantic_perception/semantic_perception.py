#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image as PILImage
import clip
import cv2
import json
import os
from datetime import datetime
import numpy as np
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import time as pytime
from sklearn.cluster import DBSCAN
import threading
from ultralytics import YOLO

class SemanticPerceptionNode(Node):
    def __init__(self):
        super().__init__('semantic_perception')
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Thread safety
        self.processing_lock = threading.Lock()
        self.processing = False

        # Camera initialization
        self.latest_depth = None
        self.camera_info = None
        self.latest_stamp = None
        self.latest_rgb = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CLIP + Faster R-CNN
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
        
        # Load YOLO12 model
        self.get_logger().info("Loading YOLO12 model...")
        self.yolo_model = YOLO("yolo12n.pt")  # Use nano version for speed
        self.get_logger().info("YOLO12 model loaded successfully!")

        # Enhanced labels for semantic understanding
        self.object_labels = ["desk", "chair", "bookshelf", "sofa", "refrigerator", "water dispenser", "toilet", "side table"]
        
        # Room/scene classification labels
        self.scene_labels = ["main office room", "break room", "bathroom"]
        
        # Pre-compute text features for efficiency
        self.get_logger().info("Pre-computing CLIP text features...")
        self.object_text_features = self._precompute_text_features(self.object_labels)
        self.scene_text_features = self._precompute_text_features(self.scene_labels)
        
        # Enhanced semantic memory
        self.semantic_memory_file = self._get_next_memory_filename()
        self.semantic_memory = self.load_semantic_memory()
        self.observation_buffer = []  # Buffer for spatial clustering
        
        # Processing control
        self.last_processing_time = 0
        self.processing_interval = 0.5  # Process every 0.5 seconds
        
        # Subscribers & Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/object_in_odom", 10)
        self.semantic_pose_pub = self.create_publisher(PoseStamped, "/semantic_location", 10)
        
        self.rgb_sub = self.create_subscription(
            Image, "/intel_realsense_r200_rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/intel_realsense_r200_depth/depth/image_raw", self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/intel_realsense_r200_depth/camera_info", self.camera_info_callback, 10)
        self.annotated_pub = self.create_publisher(Image, "/percep/annotated_image", 10)
        
        # Timer for periodic semantic analysis
        self.semantic_analysis_timer = self.create_timer(2.0, self.analyze_semantic_clusters)
        
        self.get_logger().info("Semantic Perception Node initialized successfully!")
    
    def _get_next_memory_filename(self):
        """Generate next numbered semantic memory filename"""
        base_name = "semantic_exploration_memory"
        extension = ".json"
        
        # Find the highest existing number
        max_num = 0
        import glob
        
        # Look for existing files with pattern semantic_exploration_memory_*.json
        pattern = f"{base_name}_*.json"
        existing_files = glob.glob(pattern)
        
        for file in existing_files:
            try:
                # Extract number from filename like "semantic_exploration_memory_5.json"
                num_str = file.replace(f"{base_name}_", "").replace(extension, "")
                if num_str.isdigit():
                    max_num = max(max_num, int(num_str))
            except:
                continue
        
        # Also check for the unnumbered file
        if os.path.exists(f"{base_name}.json"):
            max_num = max(max_num, 0)
        
        # Generate next filename
        next_num = max_num + 1
        next_filename = f"{base_name}_{next_num}.json"
        
        self.get_logger().info(f"Using semantic memory file: {next_filename}")
        return next_filename

    def _precompute_text_features(self, labels):
        """Pre-compute CLIP text features for efficiency"""
        with torch.no_grad():
            text_tokens = clip.tokenize(labels).to(self.device)
            text_features = F.normalize(self.clip_model.encode_text(text_tokens), dim=-1)
        return text_features

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.img_frame = "realsense_depth_frame"
        # self.get_logger().info(f"Camera info received, using frame: {self.img_frame}")

    def load_semantic_memory(self):
        """Load existing semantic memory or create new structure"""
        # Always create fresh memory for new numbered file
        self.get_logger().info(f"Creating fresh semantic memory for: {self.semantic_memory_file}")
        
        # Initialize semantic memory structure
        return {
            "objects": {},  # object_label: {positions: [], confidences: [], timestamps: []}
            "scenes": {},   # scene_label: {positions: [], confidences: [], timestamps: []}
            "locations": {}, # location_id: {scene_type: str, objects: [], center: {x,y,z}, confidence: float}
            "exploration_path": [],  # Track robot's exploration path
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "file_number": self.semantic_memory_file.split('_')[-1].replace('.json', ''),
                "node_version": "1.0"
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
        """Enhanced frame processing with semantic understanding using YOLO12"""
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            self.get_logger().warn("Missing data for processing")
            return

        cv_img = self.latest_rgb.copy()
        # self.get_logger().info(f"Processing image: {cv_img.shape}")
        
        # YOLO12 object detection (no need to manually resize - YOLO handles this)
        # self.get_logger().info("Running YOLO12 detection...")
        try:
            # Run YOLO12 inference
            results = self.yolo_model(cv_img, conf=0.25, verbose=False)  # confidence threshold 0.25
            
            # Extract detection results
            boxes = []
            scores = []
            
            if len(results) > 0 and results[0].boxes is not None:
                detections = results[0].boxes
                boxes = detections.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
                scores = detections.conf.cpu().numpy()
                
                self.get_logger().info(f"YOLO12 found {len(boxes)} objects")
                if len(boxes) > 0:
                    self.get_logger().info(f"Max score: {scores.max():.3f}, Min score: {scores.min():.3f}")
                else:
                    self.get_logger().info("No objects detected by YOLO12")
            else:
                self.get_logger().info("No objects detected by YOLO12")
                boxes = []
                scores = []
            
        except Exception as e:
            self.get_logger().error(f"YOLO12 detection error: {e}")
            boxes = []
            scores = []

        if len(boxes) == 0:
            self.get_logger().info("No objects detected")
            self._publish_annotated_image(cv_img, [], [])
            return

        # Convert to tensor format for consistency with existing code
        boxes = torch.tensor(boxes)
        detection_scores = torch.tensor(scores)
        
        self.get_logger().info(f"Processing {len(boxes)} detections")

        # Scene classification for the whole image
        # FIX: Use cv_img instead of cv_img_resized
        scene_label, scene_confidence = self._classify_scene(cv_img)
        
        self.get_logger().info(f"Scene classification: {scene_label} (confidence: {scene_confidence:.3f})")

        
        # Object classification using CLIP
        detected_objects = []
        valid_boxes = []
        
        for i, box in enumerate(boxes):
            object_label, object_confidence = self._classify_object_region(cv_img, box)
            
            if object_confidence > 0.15:  # Threshold for object detection
                # Get 3D position
                position_3d = self._get_3d_position(box, self.latest_depth)
                
                if position_3d is not None:
                    detection_data = {
                        'type': 'object',
                        'label': object_label,
                        'confidence': object_confidence * detection_scores[i].item(),  # Combined confidence
                        'position': position_3d,
                        'timestamp': datetime.now().isoformat(),
                        'box': box.tolist()
                    }
                    detected_objects.append(detection_data)
                    valid_boxes.append(box)
                    
                    # Update object memory
                    self._update_object_memory(object_label, detection_data)

        # Add scene detection to buffer
        if scene_confidence > 0.1 and len(detected_objects) > 0:
            # Use centroid of detected objects as scene center
            scene_center = self._calculate_scene_center(detected_objects)
            if scene_center is not None:
                scene_data = {
                    'type': 'scene',
                    'label': scene_label,
                    'confidence': scene_confidence,
                    'position': scene_center,
                    'timestamp': datetime.now().isoformat(),
                    'objects': [obj['label'] for obj in detected_objects]
                }
                self._update_scene_memory(scene_label, scene_data)
                self.observation_buffer.append(scene_data)

        # Add individual objects to buffer
        self.observation_buffer.extend(detected_objects)

        # Publish results
        self._publish_detections(detected_objects, scene_label if scene_confidence > 0.3 else None)
        
        # Create visualization
        labels = [f"{obj['label']} ({obj['confidence']:.2f})" for obj in detected_objects]
        if scene_confidence > 0.3:
            labels.insert(0, f"SCENE: {scene_label} ({scene_confidence:.2f})")
        
        self._publish_annotated_image(cv_img, valid_boxes, labels)
        
        # Log detections
        if detected_objects:
            obj_names = [obj['label'] for obj in detected_objects]
            self.get_logger().info(f"Detected: {obj_names} in {scene_label if scene_confidence > 0.3 else 'unknown scene'}")

        # Clear processed frames
        self.latest_rgb = None
        self.latest_depth = None

    def _classify_scene(self, image):
        """Classify the overall scene/room type"""
        try:
            # Preprocess image for CLIP
            image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = F.normalize(self.clip_model.encode_image(image_input), dim=-1)
                similarities = F.cosine_similarity(image_features, self.scene_text_features)
                
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            
            return self.scene_labels[best_idx], best_score
            
        except Exception as e:
            self.get_logger().error(f"Scene classification error: {e}")
            return "unknown", 0.0

    def _classify_object_region(self, image, box):
        """Classify object in a specific region using CLIP"""
        try:
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Ensure box is within image bounds
            h, w = image.shape[:2]
            x_min = max(0, min(x_min, w-1))
            y_min = max(0, min(y_min, h-1))
            x_max = max(x_min+1, min(x_max, w))
            y_max = max(y_min+1, min(y_max, h))
            
            # Extract and preprocess region
            region = image[y_min:y_max, x_min:x_max]
            region_pil = PILImage.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            region_input = self.clip_preprocess(region_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                region_features = F.normalize(self.clip_model.encode_image(region_input), dim=-1)
                similarities = F.cosine_similarity(region_features, self.object_text_features)
                
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            
            return self.object_labels[best_idx], best_score
            
        except Exception as e:
            self.get_logger().error(f"Object classification error: {e}")
            return "unknown", 0.0

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
        """Calculate center position of detected objects for scene localization"""
        if not detected_objects:
            return None
            
        positions = [obj['position'] for obj in detected_objects if obj.get('position')]
        if not positions:
            return None
            
        center_x = sum(pos['x'] for pos in positions) / len(positions)
        center_y = sum(pos['y'] for pos in positions) / len(positions)
        center_z = sum(pos['z'] for pos in positions) / len(positions)
        
        return {"x": center_x, "y": center_y, "z": center_z}

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
        """Update scene classification memory"""
        if label not in self.semantic_memory["scenes"]:
            self.semantic_memory["scenes"][label] = {
                "positions": [],
                "confidences": [],
                "timestamps": [],
                "object_associations": [],
                "highest_confidence": 0.0,
                "best_position": None
            }
        
        memory = self.semantic_memory["scenes"][label]
        memory["positions"].append(scene_data["position"])
        memory["confidences"].append(scene_data["confidence"])
        memory["timestamps"].append(scene_data["timestamp"])
        memory["object_associations"].append(scene_data["objects"])
        
        # Update best detection
        if scene_data["confidence"] > memory["highest_confidence"]:
            memory["highest_confidence"] = scene_data["confidence"]
            memory["best_position"] = scene_data["position"]

    def analyze_semantic_clusters(self):
        """Periodic analysis to create semantic location clusters"""
        if len(self.observation_buffer) < 5:  # Reduced threshold
            return
            
        try:
            # Extract positions for clustering
            positions = []
            observation_data = []
            
            for obs in self.observation_buffer:
                if obs.get('position'):
                    positions.append([obs['position']['x'], obs['position']['y']])
                    observation_data.append(obs)
            
            if len(positions) < 3:
                return
                
            # More aggressive spatial clustering to reduce duplicates
            positions_np = np.array(positions)
            clustering = DBSCAN(eps=5.0, min_samples=2).fit(positions_np)  # Increased eps from 2.0 to 3.0
            
            # Process clusters
            unique_labels = set(clustering.labels_)
            
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Noise
                    continue
                    
                cluster_mask = clustering.labels_ == cluster_id
                cluster_observations = [observation_data[i] for i in range(len(observation_data)) if cluster_mask[i]]
                
                # Check for similar existing locations before creating new one
                if not self._is_duplicate_location(cluster_observations):
                    self._create_semantic_location(cluster_id, cluster_observations)
            
            # Clear processed observations
            self.observation_buffer = []
            
            # Save updated memory
            self.save_semantic_memory()
            
        except Exception as e:
            self.get_logger().error(f"Semantic clustering error: {e}")
    
    def _is_duplicate_location(self, new_observations):
        """Check if this cluster is too similar to existing locations"""
        # Calculate cluster center
        positions = [obs['position'] for obs in new_observations if obs.get('position')]
        if not positions:
            return True
            
        new_center_x = sum(pos['x'] for pos in positions) / len(positions)
        new_center_y = sum(pos['y'] for pos in positions) / len(positions)
        
        # Check against existing locations
        for loc_id, loc_data in self.semantic_memory["locations"].items():
            existing_center = loc_data["center"]
            distance = np.sqrt((new_center_x - existing_center["x"])**2 + 
                            (new_center_y - existing_center["y"])**2)
            
            # If within 2 meters of existing location, consider it duplicate
            if distance < 2.0:
                # Update existing location instead
                self._update_existing_location(loc_id, new_observations)
                return True
        
        return False

    def _update_existing_location(self, location_id, new_observations):
        """Update an existing location with new observations"""
        loc_data = self.semantic_memory["locations"][location_id]
        
        # Add new objects to existing location
        new_objects = [obs['label'] for obs in new_observations if obs['type'] == 'object']
        existing_objects = set(loc_data["objects"])
        
        for obj in new_objects:
            if obj not in existing_objects:
                loc_data["objects"].append(obj)
        
        # Update timestamp and observation count
        loc_data["last_updated"] = datetime.now().isoformat()
        loc_data["observation_count"] += len(new_observations)
        
        self.get_logger().info(f"Updated existing location {location_id} with objects: {new_objects}")

    def _infer_scene_from_objects(self, detected_objects):
        """Infer scene type based on detected objects using your 3 scene labels"""
        object_labels = [obj['label'] for obj in detected_objects]
        
        # Object-to-scene mapping for your specific labels
        scene_hints = {
            "main office room": ["desk", "chair", "bookshelf", "side table"],
            "break room": ["sofa", "water dispenser", "refrigerator"],  
            "bathroom": ["toilet"]
        }
        
        scene_scores = {}
        for scene, keywords in scene_hints.items():
            score = sum(1 for obj in object_labels if obj in keywords)
            if score > 0:
                scene_scores[scene] = score / len(keywords)  # Normalize by number of keywords
        
        if scene_scores:
            best_scene = max(scene_scores.keys(), key=lambda x: scene_scores[x])
            confidence = scene_scores[best_scene]
            return best_scene, confidence
        
        return "unknown", 0.0

    def _create_semantic_location(self, cluster_id, observations):
        """Create or update a semantic location from clustered observations"""
        # Calculate cluster center
        positions = [obs['position'] for obs in observations]
        center_x = sum(pos['x'] for pos in positions) / len(positions)
        center_y = sum(pos['y'] for pos in positions) / len(positions)
        center_z = sum(pos['z'] for pos in positions) / len(positions)
        
        # Get detected objects
        object_obs = [obs for obs in observations if obs['type'] == 'object']
        objects_found = list(set(obs['label'] for obs in object_obs))
        
        # Try object-based scene inference first
        inferred_scene, inferred_confidence = self._infer_scene_from_objects(object_obs)
        
        # Get CLIP scene classification
        scene_obs = [obs for obs in observations if obs['type'] == 'scene']
        if scene_obs:
            scene_confidences = {}
            for obs in scene_obs:
                label = obs['label']
                if label not in scene_confidences:
                    scene_confidences[label] = []
                scene_confidences[label].append(obs['confidence'])
            
            # Get scene with highest average confidence
            clip_scene = max(scene_confidences.keys(), 
                            key=lambda x: sum(scene_confidences[x]) / len(scene_confidences[x]))
            clip_confidence = sum(scene_confidences[clip_scene]) / len(scene_confidences[clip_scene])
        else:
            clip_scene = "unknown"
            clip_confidence = 0.0
        
        # Choose best scene classification
        if inferred_confidence > 0.3 and inferred_confidence > clip_confidence:
            best_scene = inferred_scene
            avg_confidence = inferred_confidence
            self.get_logger().info(f"Using object-based scene inference: {best_scene}")
        elif clip_confidence > 0.2:
            best_scene = clip_scene
            avg_confidence = clip_confidence
            self.get_logger().info(f"Using CLIP scene classification: {best_scene}")
        else:
            best_scene = "unknown"
            avg_confidence = 0.0
        
        # Create location entry
        location_id = f"loc_{len(self.semantic_memory['locations'])}"
        
        location_data = {
            "scene_type": best_scene,
            "objects": objects_found,
            "center": {"x": center_x, "y": center_y, "z": center_z},
            "confidence": avg_confidence,
            "last_updated": datetime.now().isoformat(),
            "observation_count": len(observations),
            "inference_method": "object_based" if inferred_confidence > clip_confidence else "clip_based"
        }
        
        self.semantic_memory["locations"][location_id] = location_data
        
        # Publish semantic location
        self._publish_semantic_location(location_data)
        
        self.get_logger().info(f"Created semantic location: {best_scene} with objects {objects_found}")

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

    def _publish_semantic_location(self, location_data):
        """Publish semantic location"""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.position.x = location_data['center']['x']
        pose.pose.position.y = location_data['center']['y']
        pose.pose.position.z = location_data['center']['z']
        pose.pose.orientation.w = 1.0
        
        self.semantic_pose_pub.publish(pose)

    def _publish_annotated_image(self, image, boxes, labels):
        """Publish annotated image with detections"""
        annotated_img = image.copy()
        
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            if i < len(labels):
                # Handle scene label differently
                color = (255, 0, 0) if labels[i].startswith("SCENE:") else (0, 255, 0)
                cv2.putText(annotated_img, labels[i], (x_min, y_min-10), 
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