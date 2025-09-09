#!/usr/bin/env python3
import json
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from ament_index_python.packages import get_package_share_directory
from copy import deepcopy
import openai
from difflib import SequenceMatcher

class SemanticNavNode(Node):
    def __init__(self, semantic_memory_path: str = None):
        super().__init__('semantic_nav_node')

        # Initialize OpenAI for natural language understanding
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Load semantic memory from perception node
        self.semantic_memory_path = semantic_memory_path
        self.semantic_memory = self.load_semantic_memory()
        
        # Initialize navigator
        self.navigator = BasicNavigator()
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Semantic Navigator ready with perception data")

    def load_semantic_memory(self):
        """Load the latest semantic memory file from config directory"""
        if self.semantic_memory_path and os.path.exists(self.semantic_memory_path):
            with open(self.semantic_memory_path, 'r') as f:
                return json.load(f)
        else:
            # Find the latest semantic memory file in config directory
            memory_files = []
            config_dir = "config"
            
            # Check if config directory exists
            if not os.path.exists(config_dir):
                self.get_logger().warn(f"Config directory '{config_dir}' not found!")
                return {"objects": {}, "scenes": {}}
            
            # Look for semantic memory files in config directory
            for file in os.listdir(config_dir):
                if file.startswith('semantic_exploration_memory_') and file.endswith('.json'):
                    memory_files.append(os.path.join(config_dir, file))
            
            if memory_files:
                latest_file = max(memory_files, key=os.path.getmtime)
                self.get_logger().info(f"Loading semantic memory from: {latest_file}")
                with open(latest_file, 'r') as f:
                    return json.load(f)
            else:
                self.get_logger().warn(f"No semantic memory files found in '{config_dir}' directory!")
                return {"objects": {}, "scenes": {}}

    def query_with_ai(self, user_input: str):
        """Use OpenAI to understand user intent and find best match"""
        # Get available objects and scene types (no more locations)
        objects = list(self.semantic_memory.get("objects", {}).keys())
        scene_types = list(self.semantic_memory.get("scenes", {}).keys())
        
        prompt = f"""
User wants to navigate to: "{user_input}"

Available options:
OBJECTS: {objects}
SCENE TYPES: {scene_types}

Analyze the user input and determine the best navigation target. Respond with JSON:
{{
    "target_type": "object|scene",
    "target_name": "exact_name_from_available_options",
    "confidence": 0.0-1.0,
    "reasoning": "why this target was chosen"
}}

If the user asks for a scene type (like "bathroom" or "break room"), set target_type to "scene".
If they ask for a specific object (like "refrigerator" or "toilet"), set target_type to "object".
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            self.get_logger().error(f"AI query failed: {e}")
            return self.fallback_query(user_input, objects, scene_types)

    def fallback_query(self, user_input: str, objects, scene_types):
        """Fallback matching using string similarity"""
        user_lower = user_input.lower()
        best_match = None
        best_score = 0.0
        best_type = None
        
        # Check objects
        for obj in objects:
            score = SequenceMatcher(None, user_lower, obj.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = obj
                best_type = "object"
        
        # Check scene types
        for scene in scene_types:
            score = SequenceMatcher(None, user_lower, scene.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = scene
                best_type = "scene"
        
        return {
            "target_type": best_type,
            "target_name": best_match,
            "confidence": best_score,
            "reasoning": f"String similarity match (score: {best_score:.2f})"
        }

    def get_navigation_target(self, query_result):
        """Get navigation coordinates based on query result"""
        target_type = query_result["target_type"]
        target_name = query_result["target_name"]
        
        if target_type == "object":
            obj_data = self.semantic_memory["objects"].get(target_name)
            if obj_data:
                pos = obj_data["best_position"]
                return pos, f"object '{target_name}'"
                
        elif target_type == "scene":
            scene_data = self.semantic_memory["scenes"].get(target_name)
            if scene_data:
                pos = scene_data["best_position"]
                return pos, f"scene '{target_name}'"
        
        return None, None

    def navigate_to_target(self, user_input: str):
        """Main navigation function"""
        # Reload semantic memory for latest data
        self.semantic_memory = self.load_semantic_memory()
        
        # Query AI to understand intent
        query_result = self.query_with_ai(user_input)
        self.get_logger().info(f"Query result: {query_result}")
        
        if query_result["confidence"] < 0.3:
            self.get_logger().warn(f"Low confidence match: {query_result}")
            return False
        
        # Get navigation target
        position, description = self.get_navigation_target(query_result)
        
        if position is None:
            self.get_logger().error(f"Could not find navigation target for: {user_input}")
            return False
        
        # Create goal pose with odom frame and zero timestamp for nav2 compatibility
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "odom"  # Keep using odom frame from perception
        goal_pose.header.stamp = rclpy.time.Time().to_msg()  # Zero timestamp
        goal_pose.pose.position.x = float(position["x"])
        goal_pose.pose.position.y = float(position["y"])
        goal_pose.pose.position.z = 0.0  # Ground level for navigation
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Navigating to {description} at ({position['x']:.2f}, {position['y']:.2f}) in odom frame")
        
        # Start navigation
        self.navigator.goToPose(goal_pose)
        
        # Wait for navigation to complete
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f"Navigation in progress...")

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"Successfully navigated to {description}")
            return True
        elif result == TaskResult.CANCELED:
            self.get_logger().info("Navigation was canceled")
            return False
        elif result == TaskResult.FAILED:
            self.get_logger().info("Navigation failed")
            return False

    def check_navigation_setup(self):
        """Check if navigation system is properly configured"""
        self.get_logger().info("=== NAVIGATION SETUP CHECK ===")
        
        # Check if Nav2 is active
        if self.navigator.isTaskComplete():
            self.get_logger().info("✅ Nav2 system is ready")
        else:
            self.get_logger().warn("⚠️  Nav2 system status unknown")
        
        # Check semantic memory (updated for simplified structure)
        objects_count = len(self.semantic_memory.get("objects", {}))
        scenes_count = len(self.semantic_memory.get("scenes", {}))
        
        self.get_logger().info(f"📍 Semantic memory: {objects_count} objects, {scenes_count} scenes")
        
        if objects_count == 0:
            self.get_logger().warn("⚠️  No objects found - make sure perception node is running and exploring")
        
        if scenes_count == 0:
            self.get_logger().warn("⚠️  No scenes found - make sure perception node has explored different room types")
        
        # Note about frame configuration
        self.get_logger().info("📋 Navigation using odom frame - ensure Nav2 global_frame is set to 'odom'")
        self.get_logger().info("📋 If you see transform errors, run SLAM (slam_toolbox) or localization (amcl)")
        
    def show_available_targets(self):
        """Show available navigation targets"""
        objects = list(self.semantic_memory.get("objects", {}).keys())
        scenes = list(self.semantic_memory.get("scenes", {}).keys())
        
        self.get_logger().info("=== AVAILABLE NAVIGATION TARGETS ===")
        self.get_logger().info(f"OBJECTS: {', '.join(objects)}")
        self.get_logger().info(f"SCENE TYPES: {', '.join(scenes)}")
        
        # Show scene details with their unique objects
        for scene_name, scene_data in self.semantic_memory.get("scenes", {}).items():
            unique_objects = scene_data.get("unique_objects", [])
            object_count = scene_data.get("object_count", 0)
            position = scene_data.get("best_position", {})
            
            if position:
                self.get_logger().info(f"SCENE '{scene_name}': {object_count} objects ({', '.join(unique_objects)}) at ({position.get('x', 0):.2f}, {position.get('y', 0):.2f})")
            else:
                self.get_logger().info(f"SCENE '{scene_name}': {object_count} objects ({', '.join(unique_objects)}) - no position computed")
            
            



def main():
    rclpy.init()
    
    # You can specify a specific semantic memory file or let it find the latest
    node = SemanticNavNode()
    
    # Check navigation setup
    node.check_navigation_setup()
    
    # Show available targets
    node.show_available_targets()
    
    try:
        while True:
            # Get user input
            user_input = input("\nWhere do you want to go? (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Navigate to target
            success = node.navigate_to_target(user_input)
            
            if success:
                print("✅ Navigation completed successfully!")
            else:
                print("❌ Navigation failed")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()

