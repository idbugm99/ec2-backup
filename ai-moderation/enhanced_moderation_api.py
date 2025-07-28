#!=/usr/bin/env python3
"""
Enhanced Content Moderation API - NudeNet + MediaPipe Pose Analysis
Combines body part detection with pose analysis for context-aware moderation
"""

import os
import json
import time
import logging
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nudenet import NudeDetector
from PIL import Image
import werkzeug

# Import our pose analyzer
from mediapipe_pose_integration import PoseAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedModerationResult:
    """Enhanced data class for moderation results with pose analysis"""
    image_path: str
    context_type: str
    model_id: int
    nudity_score: float
    detected_parts: Dict[str, float]
    pose_analysis: Dict
    final_risk_score: float
    generated_caption: str
    policy_violations: List[str]
    moderation_status: str
    human_review_required: bool
    confidence_score: float

class EnhancedModerationAPI:
    """Enhanced moderation API with pose analysis"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
        self.upload_folder = '/tmp/nudenet_uploads'
        
        # Initialize AI models
        logger.info('Initializing NudeNet detector...')
        self.detector = NudeDetector()
        
        logger.info('Initializing MediaPipe pose analyzer...')
        self.pose_analyzer = PoseAnalyzer()
        
        # Setup routes
        self.setup_routes()
        
        # Create upload directory
        os.makedirs(self.upload_folder, exist_ok=True)
        
        logger.info('Enhanced moderation API initialized successfully')
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_upload():
            """Analyze uploaded image with both nudity and pose detection"""
            try:
                # Validate request
                if 'image' not in request.files:
                    return jsonify({'error': 'No image file provided'}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Get optional parameters
                context_type = request.form.get('context_type', 'public_site')
                model_id = int(request.form.get('model_id', 1))
                
                # Save uploaded file
                filename = f'upload_{int(time.time())}_{file.filename}'
                file_path = os.path.join(self.upload_folder, filename)
                file.save(file_path)
                
                logger.info(f'Analyzing uploaded file: {filename}')
                
                # Perform enhanced analysis
                result = self.analyze_image_enhanced(file_path, context_type, model_id)
                
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f'Analysis error: {e}')
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'services': {
                    'nudenet': True,
                    'mediapipe_pose': True
                },
                'timestamp': time.time()
            })
    
    def analyze_image_enhanced(self, image_path: str, context_type: str, model_id: int) -> Dict:
        """Perform enhanced analysis with both nudity and pose detection"""
        try:
            logger.info(f'Starting enhanced analysis for {image_path}')
            
            # Step 1: NudeNet body part detection
            logger.info('Running NudeNet detection...')
            nudenet_results = self.detector.detect(image_path)
            
            # Process NudeNet results
            detected_parts = {}
            nudity_score = 0.0
            
            for detection in nudenet_results:
                part_name = detection['class']
                confidence = detection['score']
                detected_parts[part_name] = confidence
                
                # Calculate overall nudity score
                if part_name in ['EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F', 'EXPOSED_ANUS', 'EXPOSED_BUTTOCKS']:
                    nudity_score = max(nudity_score, confidence)
            
            logger.info(f'NudeNet detected parts: {list(detected_parts.keys())}')
            
            # Step 2: MediaPipe pose analysis
            logger.info('Running pose analysis...')
            pose_analysis = self.pose_analyzer.analyze_pose(image_path)
            
            logger.info(f'Pose analysis result: {pose_analysis.get("pose_category", "unknown")}')
            
            # Step 3: Combine results for final risk assessment
            final_assessment = self.combine_analysis_results(detected_parts, pose_analysis, context_type)
            
            # Step 4: Generate comprehensive result
            result = {
                'success': True,
                'image_analysis': {
                    'nudity_detection': {
                        'detected_parts': detected_parts,
                        'nudity_score': nudity_score,
                        'has_nudity': nudity_score > 0.3
                    },
                    'pose_analysis': pose_analysis,
                    'combined_assessment': final_assessment
                },
                'moderation_decision': self.make_moderation_decision(final_assessment),
                'metadata': {
                    'context_type': context_type,
                    'model_id': model_id,
                    'analysis_timestamp': time.time(),
                    'analysis_version': '2.0_enhanced'
                }
            }
            
            logger.info(f'Analysis complete. Final risk score: {final_assessment["final_risk_score"]:.2f}')
            return result
            
        except Exception as e:
            logger.error(f'Enhanced analysis failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'context_type': context_type,
                    'model_id': model_id,
                    'analysis_timestamp': time.time()
                }
            }
    
    def combine_analysis_results(self, detected_parts: Dict, pose_analysis: Dict, context_type: str) -> Dict:
        """Combine NudeNet and pose analysis for final risk assessment"""
        
        # Base risk from nudity detection
        base_nudity_risk = 0.0
        for part, confidence in detected_parts.items():
            if part in ['EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F']:
                base_nudity_risk = max(base_nudity_risk, confidence)
            elif part in ['EXPOSED_BUTTOCKS', 'EXPOSED_ANUS']:
                base_nudity_risk = max(base_nudity_risk, confidence * 0.8)  # Slightly lower weight
        
        # Pose risk modifier
        pose_risk_modifier = 0.0
        pose_category = pose_analysis.get('pose_category', 'artistic_or_neutral')
        
        if pose_category == 'highly_suggestive':
            pose_risk_modifier = 0.4
        elif pose_category == 'moderately_suggestive':
            pose_risk_modifier = 0.2
        
        # Context modifier
        context_modifier = self.get_context_modifier(context_type)
        
        # Calculate final risk score
        final_risk_score = min(1.0, base_nudity_risk + pose_risk_modifier + context_modifier)
        
        # Determine risk level
        if final_risk_score >= 0.8:
            risk_level = 'high'
        elif final_risk_score >= 0.5:
            risk_level = 'medium'
        elif final_risk_score >= 0.3:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        # Generate reasoning
        reasoning = []
        if base_nudity_risk > 0.3:
            reasoning.append(f'Nudity detected (score: {base_nudity_risk:.2f})')
        if pose_risk_modifier > 0:
            reasoning.append(f'Suggestive pose ({pose_category})')
        if context_modifier != 0:
            reasoning.append(f'Context consideration ({context_type})')
        
        return {
            'final_risk_score': final_risk_score,
            'risk_level': risk_level,
            'base_nudity_risk': base_nudity_risk,
            'pose_risk_modifier': pose_risk_modifier,
            'context_modifier': context_modifier,
            'reasoning': reasoning
        }
    
    def get_context_modifier(self, context_type: str) -> float:
        """Get risk modifier based on usage context"""
        modifiers = {
            'public_site': 0.1,    # Higher standards for public content
            'paysite': -0.1,       # More lenient for paid content
            'private': -0.2,       # Most lenient for private content
            'store': 0.05          # Moderate standards for store content
        }
        return modifiers.get(context_type, 0.0)
    
    def make_moderation_decision(self, assessment: Dict) -> Dict:
        """Make final moderation decision based on assessment"""
        risk_score = assessment['final_risk_score']
        risk_level = assessment['risk_level']
        
        if risk_score >= 0.8:
            status = 'rejected'
            action = 'block_content'
            human_review = False
        elif risk_score >= 0.5:
            status = 'flagged_for_review'
            action = 'require_human_review'
            human_review = True
        elif risk_score >= 0.3:
            status = 'approved_with_blur'
            action = 'apply_blur_filter'
            human_review = False
        else:
            status = 'approved'
            action = 'allow_content'
            human_review = False
        
        return {
            'status': status,
            'action': action,
            'human_review_required': human_review,
            'confidence': 1.0 - abs(0.5 - risk_score),  # Higher confidence at extremes
            'suggested_blur_areas': self.suggest_blur_areas(assessment) if action == 'apply_blur_filter' else []
        }
    
    def suggest_blur_areas(self, assessment: Dict) -> List[str]:
        """Suggest which areas should be blurred"""
        # This would integrate with your existing blur coordinate system
        # For now, return general suggestions
        suggestions = []
        
        if assessment['base_nudity_risk'] > 0.3:
            suggestions.extend(['breast', 'genitalia', 'buttocks'])
        
        return suggestions
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f'Starting enhanced moderation API on {host}:{port}')
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    api = EnhancedModerationAPI()
    api.run(debug=True)
