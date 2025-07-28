#!/usr/bin/env python3
"""
Simple NudeNet-only AI Content Moderation Service
For MuseNest content moderation with minimal dependencies
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from flask import Flask, request, jsonify
from flask_cors import CORS
from nudenet import NudeDetector
from PIL import Image
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModerationResult:
    """Data class for moderation results"""
    image_path: str
    context_type: str
    model_id: int
    nudity_score: float
    detected_parts: Dict[str, float]
    pose_classification: str
    explicit_pose_score: float
    generated_caption: str
    policy_violations: List[str]
    moderation_status: str
    human_review_required: bool
    confidence_score: float

class SimpleContentModerator:
    """Simple NudeNet-based content moderation system"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize NudeNet detector
        try:
            logger.info("Loading NudeNet model...")
            self.nude_detector = NudeDetector()
            logger.info("NudeNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NudeNet model: {e}")
            self.nude_detector = None
        
        # Moderation rules (same as test version)
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
    
    def analyze_nudity(self, image_path: str) -> Tuple[float, Dict[str, float]]:
        """Analyze nudity using NudeNet"""
        try:
            if not self.nude_detector:
                # Fallback to mock data if model not loaded
                return 25.0, {'breast': 20.0, 'buttocks': 5.0}
            
            # Detect nudity
            detections = self.nude_detector.detect(image_path)
            
            # Calculate overall nudity score and part-specific scores
            part_scores = {}
            total_score = 0.0
            
            for detection in detections:
                class_name = detection['class']
                confidence = detection['score'] * 100  # Convert to percentage
                
                # Map classes to our part names
                if class_name in ['FEMALE_BREAST_EXPOSED', 'MALE_BREAST_EXPOSED']:
                    part_scores['breast'] = max(part_scores.get('breast', 0), confidence)
                elif class_name in ['BUTTOCKS_EXPOSED']:
                    part_scores['buttocks'] = max(part_scores.get('buttocks', 0), confidence)
                elif class_name in ['FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED']:
                    part_scores['genitalia'] = max(part_scores.get('genitalia', 0), confidence)
                elif class_name in ['ANUS_EXPOSED']:
                    part_scores['anus'] = max(part_scores.get('anus', 0), confidence)
                
                total_score = max(total_score, confidence)
            
            return total_score, part_scores
            
        except Exception as e:
            logger.error(f"Error analyzing nudity: {e}")
            # Return conservative estimate on error
            return 50.0, {'unknown': 50.0}
    
    def analyze_pose(self, context_type: str) -> Tuple[str, float]:
        """Simple pose analysis based on context (placeholder)"""
        # Without MediaPipe, we'll use simple heuristics based on nudity detection
        pose_mappings = {
            'profile_pic': ('portrait', 0.0),
            'public_gallery': ('artistic', 15.0),
            'premium_gallery': ('suggestive', 45.0),
            'private_content': ('explicit', 75.0)
        }
        return pose_mappings.get(context_type, ('unknown', 25.0))
    
    def generate_caption(self, context_type: str) -> str:
        """Simple caption generation based on context (placeholder)"""
        captions = {
            'profile_pic': 'A profile photograph',
            'public_gallery': 'An artistic photograph',
            'premium_gallery': 'A suggestive artistic photograph',
            'private_content': 'An adult content photograph'
        }
        return captions.get(context_type, 'A photograph')
    
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
        confidence = 0.9 if self.nude_detector else 0.6  # Lower confidence without full AI
        
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
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'version': 'nudenet-simple',
                'models_loaded': self.nude_detector is not None
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
                
                # Perform analysis
                result = self.moderate_image(image_path, context_type, model_id)
                
                if result:
                    return jsonify({
                        'success': True,
                        'result': asdict(result)
                    })
                else:
                    return jsonify({'error': 'Failed to analyze image'}), 500
                    
            except Exception as e:
                logger.error(f"Error in analyze endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/rules', methods=['GET'])
        def get_rules():
            """Get current moderation rules"""
            return jsonify({
                'success': True,
                'rules': self.moderation_rules
            })
        
        @self.app.route('/test', methods=['GET'])
        def test_all_contexts():
            """Test all context types with sample data"""
            results = {}
            
            for context in ['profile_pic', 'public_gallery', 'premium_gallery', 'private_content']:
                result = self.moderate_image(f"test_image_{context}.jpg", context, 1)
                if result:
                    results[context] = asdict(result)
            
            return jsonify({
                'success': True,
                'test_results': results
            })
    
    def moderate_image(self, image_path: str, context_type: str, model_id: int) -> Optional[ModerationResult]:
        """Main moderation function"""
        try:
            logger.info(f"Analyzing image: {image_path} for context: {context_type}")
            
            # 1. Nudity Analysis with NudeNet
            nudity_score, detected_parts = self.analyze_nudity(image_path)
            
            # 2. Simple Pose Analysis (placeholder)
            pose_class, explicit_pose_score = self.analyze_pose(context_type)
            
            # 3. Simple Caption Generation (placeholder)
            caption = self.generate_caption(context_type)
            
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
                generated_caption=caption,
                policy_violations=violations,
                moderation_status=moderation_status,
                human_review_required=human_review,
                confidence_score=confidence
            )
            
            logger.info(f"Analysis complete: {moderation_status} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in moderate_image: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Simple AI Content Moderation Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    moderator = SimpleContentModerator()
    moderator.run()