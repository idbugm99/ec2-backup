#!/usr/bin/env python3
"""
Simple test version of AI Content Moderation Service
For testing the infrastructure without heavy AI models
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from flask import Flask, request, jsonify
from flask_cors import CORS

# Mock results for testing without actual AI models
MOCK_RESULTS = {
    'nudity_analysis': {
        'profile_pic': {'score': 15.0, 'parts': {'breast': 10.0}},
        'public_gallery': {'score': 45.0, 'parts': {'breast': 40.0, 'buttocks': 15.0}},
        'premium_gallery': {'score': 75.0, 'parts': {'breast': 70.0, 'buttocks': 30.0, 'genitalia': 5.0}},
        'private_content': {'score': 95.0, 'parts': {'breast': 90.0, 'buttocks': 60.0, 'genitalia': 85.0}}
    },
    'pose_analysis': {
        'profile_pic': {'class': 'standing', 'score': 0.0},
        'public_gallery': {'class': 'artistic', 'score': 15.0},
        'premium_gallery': {'class': 'suggestive', 'score': 45.0},
        'private_content': {'class': 'explicit', 'score': 85.0}
    },
    'captions': {
        'profile_pic': 'A professional portrait of a person',
        'public_gallery': 'An artistic photo with creative lighting',
        'premium_gallery': 'A suggestive artistic photograph',
        'private_content': 'An intimate adult photograph'
    }
}

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

class TestContentModerator:
    """Test version of AI-powered content moderation system"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Moderation rules (same as production)
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
    
    def mock_nudity_analysis(self, context_type: str) -> Tuple[float, Dict[str, float]]:
        """Mock nudity analysis based on context"""
        mock_data = MOCK_RESULTS['nudity_analysis'].get(context_type, 
                    {'score': 20.0, 'parts': {'breast': 15.0}})
        return mock_data['score'], mock_data['parts']
    
    def mock_pose_analysis(self, context_type: str) -> Tuple[str, float]:
        """Mock pose analysis based on context"""
        mock_data = MOCK_RESULTS['pose_analysis'].get(context_type,
                    {'class': 'standing', 'score': 0.0})
        return mock_data['class'], mock_data['score']
    
    def mock_caption_generation(self, context_type: str) -> str:
        """Mock caption generation based on context"""
        return MOCK_RESULTS['captions'].get(context_type, 'A photograph')
    
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
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'version': 'test',
                'models_loaded': True  # Always true for test version
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
                
                # Perform mock analysis
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
            """Test all context types with mock data"""
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
        """Main moderation function (test version)"""
        try:
            logger.info(f"[TEST] Analyzing image: {image_path} for context: {context_type}")
            
            # Add slight delay to simulate processing
            time.sleep(0.1)
            
            # 1. Mock Nudity Analysis
            nudity_score, detected_parts = self.mock_nudity_analysis(context_type)
            
            # 2. Mock Pose Analysis
            pose_class, explicit_pose_score = self.mock_pose_analysis(context_type)
            
            # 3. Mock Caption Generation
            caption = self.mock_caption_generation(context_type)
            
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
            
            logger.info(f"[TEST] Analysis complete: {moderation_status} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in moderate_image: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting AI Content Moderation Test Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    moderator = TestContentModerator()
    moderator.run()