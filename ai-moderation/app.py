#!/usr/bin/env python3
"""
MuseNest AI Content Moderation Service
Multi-layer AI analysis for context-aware content moderation
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# AI Libraries
from nudenet import NudeDetector
import mediapipe as mp
from transformers import BlipProcessor, BlipForConditionalGeneration

# Database
import mysql.connector
from mysql.connector import Error

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModerationResult:
    """Data class for moderation results"""
    image_path: str
    context_type: str
    model_id: int
    
    # NudeNet Results
    nudity_score: float
    detected_parts: Dict[str, float]
    
    # Pose Analysis
    pose_classification: str
    explicit_pose_score: float
    pose_keypoints: Optional[Dict]
    
    # Content Understanding
    generated_caption: str
    policy_violations: List[str]
    
    # Final Decision
    moderation_status: str  # 'approved', 'flagged', 'rejected'
    human_review_required: bool
    confidence_score: float

class ContentModerator:
    """AI-powered content moderation system"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize AI models
        self.nude_detector = None
        self.pose_detector = None
        self.blip_processor = None
        self.blip_model = None
        
        # Database connection
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'musenest'),
            'port': int(os.getenv('DB_PORT', 3306))
        }
        
        # Moderation rules
        self.moderation_rules = {
            'profile_pic': {
                'max_nudity_score': 30,
                'allowed_poses': ['standing', 'sitting', 'portrait'],
                'banned_keywords': ['sexual', 'explicit', 'penetration']
            },
            'public_gallery': {
                'max_nudity_score': 60,
                'allowed_poses': ['artistic', 'suggestive', 'lingerie'],
                'banned_keywords': ['sexual_act', 'explicit_contact']
            },
            'premium_gallery': {
                'max_nudity_score': 90,
                'allowed_poses': ['all_except_explicit'],  
                'banned_keywords': ['illegal', 'violence']
            },
            'private_content': {
                'max_nudity_score': 100,
                'allowed_poses': ['all'],
                'banned_keywords': ['illegal', 'underage', 'violence']
            }
        }
        
        self.setup_routes()
        
    def initialize_models(self):
        """Initialize AI models (lazy loading)"""
        try:
            logger.info("Initializing AI models...")
            
            # Initialize NudeNet
            if self.nude_detector is None:
                logger.info("Loading NudeNet detector...")
                self.nude_detector = NudeDetector()
                logger.info("NudeNet loaded successfully")
            
            # Initialize MediaPipe Pose
            if self.pose_detector is None:
                logger.info("Loading MediaPipe Pose detector...")
                self.pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe Pose loaded successfully")
            
            # Initialize BLIP for image captioning
            if self.blip_processor is None or self.blip_model is None:
                logger.info("Loading BLIP model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("BLIP loaded successfully")
                
            logger.info("All AI models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def analyze_nudity(self, image_path: str) -> Tuple[float, Dict[str, float]]:
        """Analyze nudity using NudeNet"""
        try:
            # Detect nudity
            results = self.nude_detector.detect(image_path)
            
            # Parse results
            detected_parts = {}
            max_score = 0.0
            
            for result in results:
                class_name = result['class']
                score = result['score']
                
                # Map NudeNet classes to our categories
                if class_name in ['BREAST_F', 'BREAST_M']:
                    detected_parts['breast'] = max(detected_parts.get('breast', 0), score)
                elif class_name in ['BUTTOCKS_EXPOSED']:
                    detected_parts['buttocks'] = max(detected_parts.get('buttocks', 0), score)
                elif class_name in ['GENITAL_F', 'GENITAL_M']:
                    detected_parts['genitalia'] = max(detected_parts.get('genitalia', 0), score)
                
                max_score = max(max_score, score)
            
            # Convert to percentage
            nudity_score = max_score * 100
            detected_parts = {k: v * 100 for k, v in detected_parts.items()}
            
            return nudity_score, detected_parts
            
        except Exception as e:
            logger.error(f"Error in nudity analysis: {e}")
            return 0.0, {}
    
    def analyze_pose(self, image_path: str) -> Tuple[str, float, Optional[Dict]]:
        """Analyze pose using MediaPipe"""
        try:
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose_detector.process(image_rgb)
            
            if not results.pose_landmarks:
                return "no_pose_detected", 0.0, None
            
            # Extract keypoints
            keypoints = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[f"point_{idx}"] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            # Simple pose classification based on keypoint positions
            pose_class, explicit_score = self.classify_pose(keypoints)
            
            return pose_class, explicit_score, keypoints
            
        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")
            return "error", 0.0, None
    
    def classify_pose(self, keypoints: Dict) -> Tuple[str, float]:
        """Simple pose classification logic"""
        try:
            # This is a simplified version - you'd want more sophisticated logic
            # For now, we'll use basic heuristics
            
            # Get key body parts
            nose = keypoints.get('point_0', {})
            left_shoulder = keypoints.get('point_11', {})
            right_shoulder = keypoints.get('point_12', {})
            left_hip = keypoints.get('point_23', {})
            right_hip = keypoints.get('point_24', {})
            
            # Calculate basic metrics
            if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
                return "unclear", 0.0
            
            # Simple classification based on pose characteristics
            # This would be much more sophisticated in production
            shoulder_line = abs(left_shoulder.get('y', 0) - right_shoulder.get('y', 0))
            body_angle = abs(nose.get('y', 0) - (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2)
            
            if shoulder_line < 0.1 and body_angle < 0.3:
                return "standing", 0.0
            elif shoulder_line > 0.2:
                return "leaning", 25.0
            else:
                return "sitting", 0.0
                
        except Exception as e:
            logger.error(f"Error in pose classification: {e}")
            return "error", 0.0
    
    def generate_caption(self, image_path: str) -> str:
        """Generate image caption using BLIP"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt")
            
            # Generate caption
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error in caption generation: {e}")
            return "Unable to generate caption"
    
    def check_policy_violations(self, caption: str, context_type: str) -> List[str]:
        """Check for policy violations based on caption and context"""
        violations = []
        
        rules = self.moderation_rules.get(context_type, {})
        banned_keywords = rules.get('banned_keywords', [])
        
        caption_lower = caption.lower()
        
        for keyword in banned_keywords:
            if keyword in caption_lower:
                violations.append(keyword)
        
        return violations
    
    def apply_moderation_rules(self, nudity_score: float, pose_class: str, 
                             explicit_pose_score: float, violations: List[str], 
                             context_type: str) -> Tuple[str, bool, float]:
        """Apply moderation rules and make final decision"""
        
        rules = self.moderation_rules.get(context_type, {})
        max_nudity = rules.get('max_nudity_score', 0)
        allowed_poses = rules.get('allowed_poses', [])
        
        # Calculate confidence score
        confidence = 0.8  # Base confidence
        
        # Check nudity threshold
        if nudity_score > max_nudity:
            return "rejected", False, confidence
        
        # Check pose restrictions
        if allowed_poses != ['all'] and pose_class not in allowed_poses:
            if explicit_pose_score > 50:
                return "rejected", False, confidence
            elif explicit_pose_score > 25:
                return "flagged", True, confidence * 0.7
        
        # Check policy violations
        if violations:
            if len(violations) > 1:
                return "rejected", False, confidence
            else:
                return "flagged", True, confidence * 0.8
        
        # Approve if all checks pass
        if nudity_score < max_nudity * 0.5:
            return "approved", False, confidence
        else:
            return "approved", False, confidence * 0.9
    
    def save_moderation_result(self, result: ModerationResult) -> bool:
        """Save moderation result to database"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            query = """
                INSERT INTO content_moderation (
                    image_path, model_id, context_type, nudity_score, detected_parts,
                    pose_classification, explicit_pose_score, pose_keypoints,
                    generated_caption, policy_violations, moderation_status,
                    human_review_required, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
            """
            
            values = (
                result.image_path,
                result.model_id,
                result.context_type,
                result.nudity_score,
                json.dumps(result.detected_parts),
                result.pose_classification,
                result.explicit_pose_score,
                json.dumps(result.pose_keypoints) if result.pose_keypoints else None,
                result.generated_caption,
                json.dumps(result.policy_violations),
                result.moderation_status,
                result.human_review_required
            )
            
            cursor.execute(query, values)
            connection.commit()
            
            logger.info(f"Saved moderation result for {result.image_path}")
            return True
            
        except Error as e:
            logger.error(f"Error saving to database: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'models_loaded': all([
                    self.nude_detector is not None,
                    self.pose_detector is not None,
                    self.blip_processor is not None,
                    self.blip_model is not None
                ])
            })
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_image():
            try:
                data = request.json
                image_path = data.get('image_path')
                context_type = data.get('context_type', 'public_gallery')
                model_id = data.get('model_id', 0)
                
                if not image_path:
                    return jsonify({'error': 'image_path required'}), 400
                
                # Initialize models if not already loaded
                if not self.initialize_models():
                    return jsonify({'error': 'Failed to initialize AI models'}), 500
                
                # Perform analysis
                result = self.moderate_image(image_path, context_type, model_id)
                
                if result:
                    return jsonify({
                        'success': True,
                        'result': {
                            'moderation_status': result.moderation_status,
                            'human_review_required': result.human_review_required,
                            'confidence_score': result.confidence_score,
                            'nudity_score': result.nudity_score,
                            'detected_parts': result.detected_parts,
                            'pose_classification': result.pose_classification,
                            'explicit_pose_score': result.explicit_pose_score,
                            'generated_caption': result.generated_caption,
                            'policy_violations': result.policy_violations
                        }
                    })
                else:
                    return jsonify({'error': 'Failed to analyze image'}), 500
                    
            except Exception as e:
                logger.error(f"Error in analyze endpoint: {e}")
                return jsonify({'error': str(e)}), 500
    
    def moderate_image(self, image_path: str, context_type: str, model_id: int) -> Optional[ModerationResult]:
        """Main moderation function"""
        try:
            logger.info(f"Analyzing image: {image_path} for context: {context_type}")
            
            # 1. Nudity Analysis
            nudity_score, detected_parts = self.analyze_nudity(image_path)
            
            # 2. Pose Analysis
            pose_class, explicit_pose_score, pose_keypoints = self.analyze_pose(image_path)
            
            # 3. Caption Generation
            caption = self.generate_caption(image_path)
            
            # 4. Policy Violation Check
            violations = self.check_policy_violations(caption, context_type)
            
            # 5. Apply Moderation Rules
            moderation_status, human_review, confidence = self.apply_moderation_rules(
                nudity_score, pose_class, explicit_pose_score, violations, context_type
            )
            
            # 6. Create Result Object
            result = ModerationResult(
                image_path=image_path,
                context_type=context_type,
                model_id=model_id,
                nudity_score=nudity_score,
                detected_parts=detected_parts,
                pose_classification=pose_class,
                explicit_pose_score=explicit_pose_score,
                pose_keypoints=pose_keypoints,
                generated_caption=caption,
                policy_violations=violations,
                moderation_status=moderation_status,
                human_review_required=human_review,
                confidence_score=confidence
            )
            
            # 7. Save to Database
            self.save_moderation_result(result)
            
            logger.info(f"Analysis complete: {moderation_status} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in moderate_image: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting AI Content Moderation Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    moderator = ContentModerator()
    moderator.run()