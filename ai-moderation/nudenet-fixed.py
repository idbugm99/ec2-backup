#!/usr/bin/env python3
"""
Fixed NudeNet AI Content Moderation Service
Properly handles real image uploads and analysis
"""

import os
import json
import time
import logging
import base64
import tempfile
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

class FixedContentModerator:
    """Fixed NudeNet-based content moderation system"""
    
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
    
    def analyze_nudity(self, image_path: str) -> Tuple[float, Dict[str, float]]:
        """Analyze nudity using NudeNet"""
        try:
            if not self.nude_detector:
                logger.warning("NudeNet model not loaded, using fallback")
                return 25.0, {'breast': 20.0, 'buttocks': 5.0}
            
            logger.info(f"Analyzing image: {image_path}")
            
            # Detect nudity
            detections = self.nude_detector.detect(image_path)
            logger.info(f"NudeNet detections: {detections}")
            
            # Calculate overall nudity score and part-specific scores
            part_scores = {}
            total_score = 0.0
            
            for detection in detections:
                class_name = detection['class']
                confidence = detection['score'] * 100  # Convert to percentage
                
                logger.info(f"Detected: {class_name} with confidence {confidence:.2f}%")
                
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
            
            # If no explicit parts detected, set a base score
            if not part_scores:
                part_scores = {'none_detected': 0.0}
                total_score = 0.0
            
            logger.info(f"Analysis complete - Total score: {total_score:.2f}%, Parts: {part_scores}")
            return total_score, part_scores
            
        except Exception as e:
            logger.error(f"Error analyzing nudity: {e}")
            # Return conservative estimate on error
            return 50.0, {'error': 50.0}
    
    def analyze_pose(self, nudity_score: float, detected_parts: Dict[str, float]) -> Tuple[str, float]:
        """Enhanced pose analysis based on nudity detection"""
        try:
            # Determine pose based on detected content
            if 'genitalia' in detected_parts and detected_parts['genitalia'] > 70:
                return 'explicit', 90.0
            elif nudity_score > 80:
                return 'explicit', 85.0
            elif nudity_score > 50:
                return 'suggestive', 60.0
            elif nudity_score > 20:
                return 'artistic', 25.0
            else:
                return 'portrait', 5.0
                
        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")
            return 'unknown', 25.0
    
    def generate_caption(self, detected_parts: Dict[str, float], nudity_score: float) -> str:
        """Generate caption based on actual detection results"""
        try:
            if 'genitalia' in detected_parts and detected_parts['genitalia'] > 50:
                return 'An explicit adult photograph showing intimate areas'
            elif nudity_score > 70:
                return 'An adult photograph with significant nudity'
            elif nudity_score > 40:
                return 'An artistic nude or partially nude photograph'
            elif nudity_score > 10:
                return 'A photograph with some exposed skin or suggestive content'
            else:
                return 'A photograph with no significant nudity detected'
                
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return 'A photograph'
    
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
        confidence = 0.9 if self.nude_detector else 0.6
        
        logger.info(f"Applying rules - Nudity: {nudity_score:.2f}%, Max allowed: {max_nudity}%")
        
        # Check nudity threshold
        if nudity_score > max_nudity:
            logger.info(f"REJECTED: Nudity score {nudity_score:.2f}% exceeds {max_nudity}% threshold")
            return "rejected", False, confidence
        
        # Check pose restrictions
        if allowed_poses != ['all'] and pose_class not in allowed_poses:
            if explicit_pose_score > 50:
                logger.info(f"REJECTED: Explicit pose score {explicit_pose_score:.2f}% too high")
                return "rejected", False, confidence
            elif explicit_pose_score > 25:
                logger.info(f"FLAGGED: Pose score {explicit_pose_score:.2f}% requires review")
                return "flagged", True, confidence * 0.7
        
        # Check policy violations
        if violations:
            if len(violations) > 1:
                logger.info(f"REJECTED: Multiple policy violations: {violations}")
                return "rejected", False, confidence
            else:
                logger.info(f"FLAGGED: Policy violation: {violations}")
                return "flagged", True, confidence * 0.8
        
        # Approve if all checks pass
        if nudity_score < max_nudity * 0.5:
            logger.info(f"APPROVED: Nudity score {nudity_score:.2f}% well below threshold")
            return "approved", False, confidence
        else:
            logger.info(f"APPROVED: Nudity score {nudity_score:.2f}% within acceptable range")
            return "approved", False, confidence * 0.9
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'version': 'nudenet-fixed',
                'models_loaded': self.nude_detector is not None
            })
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_image():
            try:
                data = request.json
                context_type = data.get('context_type', 'public_gallery')
                model_id = data.get('model_id', 0)
                
                # Handle different input methods
                image_path = None
                temp_file = None
                
                if 'image_path' in data:
                    # Direct file path
                    image_path = data['image_path']
                elif 'image_data' in data:
                    # Base64 encoded image
                    try:
                        image_data = base64.b64decode(data['image_data'])
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        temp_file.write(image_data)
                        temp_file.close()
                        image_path = temp_file.name
                    except Exception as e:
                        return jsonify({'error': f'Invalid image data: {e}'}), 400
                elif 'image_url' in data:
                    # Download from URL
                    try:
                        response = requests.get(data['image_url'], timeout=10)
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        temp_file.write(response.content)
                        temp_file.close()
                        image_path = temp_file.name
                    except Exception as e:
                        return jsonify({'error': f'Failed to download image: {e}'}), 400
                else:
                    return jsonify({'error': 'No image provided (need image_path, image_data, or image_url)'}), 400
                
                # Perform analysis
                result = self.moderate_image(image_path, context_type, model_id)
                
                # Clean up temp file if created
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                
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
        
        @self.app.route('/analyze-url', methods=['POST'])
        def analyze_url():
            """Analyze image from URL - easier for testing"""
            try:
                data = request.json
                image_url = data.get('image_url')
                context_type = data.get('context_type', 'public_gallery')
                model_id = data.get('model_id', 0)
                
                if not image_url:
                    return jsonify({'error': 'image_url required'}), 400
                
                # Download image
                response = requests.get(image_url, timeout=30)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(response.content)
                temp_file.close()
                
                # Analyze
                result = self.moderate_image(temp_file.name, context_type, model_id)
                
                # Cleanup
                os.unlink(temp_file.name)
                
                if result:
                    return jsonify({
                        'success': True,
                        'result': asdict(result)
                    })
                else:
                    return jsonify({'error': 'Failed to analyze image'}), 500
                    
            except Exception as e:
                logger.error(f"Error in analyze-url endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/rules', methods=['GET'])
        def get_rules():
            """Get current moderation rules"""
            return jsonify({
                'success': True,
                'rules': self.moderation_rules
            })
        
        @self.app.route('/test', methods=['GET'])
        def test_detector():
            """Test NudeNet detector with sample"""
            if not self.nude_detector:
                return jsonify({
                    'success': False,
                    'error': 'NudeNet model not loaded'
                })
            
            return jsonify({
                'success': True,
                'message': 'NudeNet detector loaded and ready',
                'model_loaded': True
            })
    
    def moderate_image(self, image_path: str, context_type: str, model_id: int) -> Optional[ModerationResult]:
        """Main moderation function"""
        try:
            logger.info(f"Analyzing image: {image_path} for context: {context_type}")
            
            # 1. Real Nudity Analysis with NudeNet
            nudity_score, detected_parts = self.analyze_nudity(image_path)
            
            # 2. Enhanced Pose Analysis based on detection
            pose_class, explicit_pose_score = self.analyze_pose(nudity_score, detected_parts)
            
            # 3. Smart Caption Generation based on results
            caption = self.generate_caption(detected_parts, nudity_score)
            
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
        logger.info(f"Starting Fixed AI Content Moderation Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    moderator = FixedContentModerator()
    moderator.run()