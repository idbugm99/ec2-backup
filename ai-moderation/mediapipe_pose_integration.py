#\!/usr/bin/env python3
"""
MediaPipe Pose Integration with NudeNet
Analyzes body positioning to distinguish artistic vs suggestive poses
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    """MediaPipe-based pose analysis for content moderation"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def analyze_pose(self, image_path: str) -> Dict:
        """Analyze pose from image and return pose classification"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return self._create_error_result('Could not read image')
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_image)
            
            if not results.pose_landmarks:
                return self._create_result('no_pose_detected', 0.0, {})
            
            # Extract pose analysis
            landmarks = results.pose_landmarks.landmark
            pose_analysis = self._analyze_landmarks(landmarks, image.shape)
            
            return pose_analysis
            
        except Exception as e:
            logger.error(f'Pose analysis error: {e}')
            return self._create_error_result(str(e))
    
    def _analyze_landmarks(self, landmarks, image_shape) -> Dict:
        """Analyze pose landmarks to determine body positioning"""
        # Get key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP] 
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Calculate body angles and positions
        torso_angle = self._calculate_torso_angle(left_shoulder, right_shoulder, left_hip, right_hip)
        hip_bend_angle = self._calculate_hip_bend_angle(left_shoulder, left_hip, left_knee)
        body_orientation = self._calculate_body_orientation(nose, left_shoulder, right_shoulder)
        leg_spread = self._calculate_leg_spread(left_hip, right_hip, left_knee, right_knee)
        
        # Classify pose based on measurements
        pose_classification = self._classify_pose(torso_angle, hip_bend_angle, body_orientation, leg_spread)
        
        return self._create_result(
            pose_classification['category'],
            pose_classification['suggestive_score'],
            {
                'torso_angle': torso_angle,
                'hip_bend_angle': hip_bend_angle, 
                'body_orientation': body_orientation,
                'leg_spread': leg_spread,
                'reasoning': pose_classification['reasoning']
            }
        )
    
    def _calculate_torso_angle(self, left_shoulder, right_shoulder, left_hip, right_hip) -> float:
        """Calculate angle of torso relative to vertical"""
        # Midpoint of shoulders and hips
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        
        # Calculate angle from vertical
        angle = math.degrees(math.atan2(abs(shoulder_mid_x - hip_mid_x), abs(shoulder_mid_y - hip_mid_y)))
        return angle
    
    def _calculate_hip_bend_angle(self, shoulder, hip, knee) -> float:
        """Calculate hip bend angle (indicates bending over)"""
        # Vector from hip to shoulder
        vec1_x = shoulder.x - hip.x
        vec1_y = shoulder.y - hip.y
        
        # Vector from hip to knee  
        vec2_x = knee.x - hip.x
        vec2_y = knee.y - hip.y
        
        # Calculate angle between vectors
        dot_product = vec1_x * vec2_x + vec1_y * vec2_y
        mag1 = math.sqrt(vec1_x**2 + vec1_y**2)
        mag2 = math.sqrt(vec2_x**2 + vec2_y**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
            
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _calculate_body_orientation(self, nose, left_shoulder, right_shoulder) -> str:
        """Determine if body is facing camera, turned away, or sideways"""
        # Calculate shoulder width in image
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        # Determine orientation based on shoulder visibility and nose position
        if shoulder_width < 0.1:  # Very narrow shoulders = turned away
            return 'turned_away'
        elif shoulder_width > 0.2:  # Wide shoulders = facing camera
            return 'facing_camera'
        else:
            return 'sideways'
    
    def _calculate_leg_spread(self, left_hip, right_hip, left_knee, right_knee) -> float:
        """Calculate leg spread distance"""
        hip_distance = abs(left_hip.x - right_hip.x)
        knee_distance = abs(left_knee.x - right_knee.x)
        
        # Return the maximum spread
        return max(hip_distance, knee_distance)
    
    def _classify_pose(self, torso_angle: float, hip_bend_angle: float, 
                      body_orientation: str, leg_spread: float) -> Dict:
        """Classify pose as artistic vs suggestive based on measurements"""
        
        suggestive_score = 0.0
        reasoning = []
        
        # Bent over poses (high suggestive potential)
        if hip_bend_angle < 100 and torso_angle > 30:
            suggestive_score += 0.4
            reasoning.append('bent_over_pose')
        
        # Legs spread wide (suggestive)
        if leg_spread > 0.3:
            suggestive_score += 0.3
            reasoning.append('wide_leg_spread')
        
        # Turned away with bent posture (showing buttocks)
        if body_orientation == 'turned_away' and hip_bend_angle < 120:
            suggestive_score += 0.4
            reasoning.append('turned_away_bent')
        
        # Extreme torso angles (arched back, etc.)
        if torso_angle > 45:
            suggestive_score += 0.2
            reasoning.append('extreme_torso_angle')
        
        # Determine category
        if suggestive_score >= 0.6:
            category = 'highly_suggestive'
        elif suggestive_score >= 0.3:
            category = 'moderately_suggestive' 
        else:
            category = 'artistic_or_neutral'
        
        return {
            'category': category,
            'suggestive_score': min(suggestive_score, 1.0),
            'reasoning': reasoning
        }
    
    def _create_result(self, category: str, score: float, details: Dict) -> Dict:
        """Create standardized pose analysis result"""
        return {
            'pose_detected': True,
            'pose_category': category,
            'suggestive_score': score,
            'details': details,
            'error': None
        }
    
    def _create_error_result(self, error_msg: str) -> Dict:
        """Create error result"""
        return {
            'pose_detected': False,
            'pose_category': 'error',
            'suggestive_score': 0.0,
            'details': {},
            'error': error_msg
        }

# Test function
if __name__ == '__main__':
    analyzer = PoseAnalyzer()
    # Test with an image
    result = analyzer.analyze_pose('/path/to/test/image.jpg')
    print(json.dumps(result, indent=2))
