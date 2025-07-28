#!/usr/bin/env python3
"""
NudeNet Upload API - Accept file uploads via REST API
Allows direct file upload for real AI analysis
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

class NudeNetUploadAPI:
    """NudeNet API with file upload support"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
        self.upload_folder = '/tmp/nudenet_uploads'
        os.makedirs(self.upload_folder, exist_ok=True)
        
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
            'profile_pic': {'max_nudity_score': 30},
            'public_gallery': {'max_nudity_score': 60},
            'premium_gallery': {'max_nudity_score': 90},
            'private_content': {'max_nudity_score': 100}
        }
        
        self.setup_routes()
    
    def analyze_nudity(self, image_path: str) -> Tuple[float, Dict[str, float]]:
        """Analyze nudity using NudeNet"""
        try:
            if not self.nude_detector:
                logger.warning("NudeNet model not loaded")
                return 0.0, {'error': 'model_not_loaded'}
            
            logger.info(f"Analyzing image: {image_path}")
            
            # Detect nudity
            detections = self.nude_detector.detect(image_path)
            logger.info(f"NudeNet detections: {detections}")
            
            # Calculate scores and locations
            part_scores = {}
            part_locations = {}
            total_score = 0.0
            
            for detection in detections:
                class_name = detection['class']
                confidence = detection['score'] * 100
                
                # Get bounding box coordinates
                bbox = detection['box']
                location = {
                    'x': int(bbox[0]),      # left
                    'y': int(bbox[1]),      # top  
                    'width': int(bbox[2]),  # width
                    'height': int(bbox[3]), # height
                    'confidence': round(confidence, 2)
                }
                
                logger.info(f"Detected: {class_name} at ({bbox[0]}, {bbox[1]}) with confidence {confidence:.2f}%")
                
                # Map classes to parts with locations
                if class_name in ['FEMALE_BREAST_EXPOSED', 'MALE_BREAST_EXPOSED']:
                    part_scores['breast'] = max(part_scores.get('breast', 0), confidence)
                    if 'breast' not in part_locations or confidence > part_locations['breast']['confidence']:
                        part_locations['breast'] = location
                elif class_name in ['BUTTOCKS_EXPOSED']:
                    part_scores['buttocks'] = max(part_scores.get('buttocks', 0), confidence)
                    if 'buttocks' not in part_locations or confidence > part_locations['buttocks']['confidence']:
                        part_locations['buttocks'] = location
                elif class_name in ['FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED']:
                    part_scores['genitalia'] = max(part_scores.get('genitalia', 0), confidence)
                    if 'genitalia' not in part_locations or confidence > part_locations['genitalia']['confidence']:
                        part_locations['genitalia'] = location
                elif class_name in ['ANUS_EXPOSED']:
                    part_scores['anus'] = max(part_scores.get('anus', 0), confidence)
                    if 'anus' not in part_locations or confidence > part_locations['anus']['confidence']:
                        part_locations['anus'] = location
                
                total_score = max(total_score, confidence)
            
            if not part_scores:
                part_scores = {'none_detected': 0.0}
                total_score = 0.0
            
            logger.info(f"Analysis complete - Total: {total_score:.2f}%, Parts: {part_scores}")
            logger.info(f"Part locations: {part_locations}")
            return total_score, part_scores, part_locations
            
        except Exception as e:
            logger.error(f"Error analyzing nudity: {e}")
            return 0.0, {'error': str(e)}, {}
    
    def analyze_pose_details(self, detected_parts: Dict[str, float], nudity_score: float) -> Tuple[str, float, str]:
        """Provide detailed pose analysis based on detected body parts"""
        
        # Extract part scores
        genitalia = detected_parts.get('genitalia', 0)
        breast = detected_parts.get('breast', 0) 
        buttocks = detected_parts.get('buttocks', 0)
        anus = detected_parts.get('anus', 0)
        
        # Detailed pose classification
        if genitalia > 60:
            if anus > 30:
                return 'explicit-penetration', 95.0, 'Explicit adult content with full genital exposure and anal visibility'
            else:
                return 'explicit-genital', 90.0, 'Explicit adult content with prominent genital exposure'
        elif genitalia > 30:
            if breast > 60:
                return 'explicit-full-nudity', 85.0, 'Full nudity with visible genitalia and breasts'
            else:
                return 'explicit-partial', 75.0, 'Adult content with partial genital exposure'
        elif breast > 70:
            if buttocks > 50:
                return 'explicit-topless-rear', 70.0, 'Topless adult content with rear nudity'
            else:
                return 'explicit-topless', 65.0, 'Topless adult content with full breast exposure'
        elif breast > 40:
            if buttocks > 40:
                return 'suggestive-semi-nude', 60.0, 'Semi-nude content with partial breast and buttocks exposure'
            else:
                return 'suggestive-topless-partial', 55.0, 'Suggestive content with partial breast exposure'
        elif buttocks > 60:
            if anus > 20:
                return 'explicit-rear-exposed', 65.0, 'Adult content with full buttocks and anal exposure'
            else:
                return 'suggestive-rear-nude', 50.0, 'Suggestive content with full buttocks nudity'
        elif buttocks > 30:
            return 'suggestive-rear-partial', 45.0, 'Suggestive content with partial buttocks exposure'
        elif breast > 20:
            return 'suggestive-breast-partial', 40.0, 'Suggestive content with minor breast exposure'
        elif nudity_score > 10:
            return 'artistic-implied', 25.0, 'Artistic content with implied nudity or minimal exposure'
        else:
            return 'safe-portrait', 5.0, 'Safe content with no significant nudity detected'
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'version': 'nudenet-upload-api',
                'models_loaded': self.nude_detector is not None,
                'upload_enabled': True
            })
        
        @self.app.route('/upload', methods=['POST'])
        def upload_and_analyze():
            """Upload file and analyze with NudeNet"""
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Get parameters
                context_type = request.form.get('context_type', 'public_gallery')
                model_id = int(request.form.get('model_id', 1))
                
                # Validate file type
                if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, GIF, or BMP'}), 400
                
                # Save uploaded file
                filename = werkzeug.utils.secure_filename(file.filename)
                timestamp = int(time.time() * 1000)
                safe_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(self.upload_folder, safe_filename)
                
                file.save(filepath)
                logger.info(f"File saved: {filepath}")
                
                # Analyze with NudeNet
                result = self.moderate_image(filepath, context_type, model_id)
                
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except:
                    pass
                
                if result:
                    # Convert result object to dict
                    result_dict = {
                        'image_path': result.image_path,
                        'context_type': result.context_type,
                        'model_id': result.model_id,
                        'nudity_score': result.nudity_score,
                        'detected_parts': result.detected_parts,
                        'part_locations': result.part_locations,
                        'pose_classification': result.pose_classification,
                        'explicit_pose_score': result.explicit_pose_score,
                        'generated_caption': result.generated_caption,
                        'policy_violations': result.policy_violations,
                        'moderation_status': result.moderation_status,
                        'human_review_required': result.human_review_required,
                        'confidence_score': result.confidence_score
                    }
                    
                    return jsonify({
                        'success': True,
                        'result': result_dict
                    })
                else:
                    return jsonify({'error': 'Analysis failed'}), 500
                    
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_existing():
            """Analyze existing file path"""
            try:
                data = request.json
                image_path = data.get('image_path')
                context_type = data.get('context_type', 'public_gallery')
                model_id = data.get('model_id', 1)
                
                if not image_path:
                    return jsonify({'error': 'image_path required'}), 400
                
                result = self.moderate_image(image_path, context_type, model_id)
                
                if result:
                    # Convert result object to dict
                    result_dict = {
                        'image_path': result.image_path,
                        'context_type': result.context_type,
                        'model_id': result.model_id,
                        'nudity_score': result.nudity_score,
                        'detected_parts': result.detected_parts,
                        'part_locations': result.part_locations,
                        'pose_classification': result.pose_classification,
                        'explicit_pose_score': result.explicit_pose_score,
                        'generated_caption': result.generated_caption,
                        'policy_violations': result.policy_violations,
                        'moderation_status': result.moderation_status,
                        'human_review_required': result.human_review_required,
                        'confidence_score': result.confidence_score
                    }
                    
                    return jsonify({
                        'success': True,
                        'result': result_dict
                    })
                else:
                    return jsonify({'error': 'Analysis failed'}), 500
                    
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def moderate_image(self, image_path: str, context_type: str, model_id: int) -> Optional[ModerationResult]:
        """Main moderation function"""
        try:
            logger.info(f"Analyzing: {image_path} for context: {context_type}")
            
            # 1. Real Nudity Analysis
            nudity_score, detected_parts, part_locations = self.analyze_nudity(image_path)
            
            # 2. Generate detailed pose analysis
            pose_class, explicit_score, caption = self.analyze_pose_details(detected_parts, nudity_score)
            
            # 3. Apply moderation rules
            threshold = self.moderation_rules.get(context_type, {}).get('max_nudity_score', 60)
            
            if nudity_score > threshold:
                status, review_needed = 'rejected', False
            elif nudity_score > threshold * 0.8:
                status, review_needed = 'flagged', True
            else:
                status, review_needed = 'approved', False
            
            # 4. Create result with locations
            result_dict = {
                'image_path': image_path,
                'context_type': context_type,
                'model_id': model_id,
                'nudity_score': nudity_score,
                'detected_parts': detected_parts,
                'part_locations': part_locations,  # NEW: Bounding box coordinates
                'pose_classification': pose_class,
                'explicit_pose_score': explicit_score,
                'generated_caption': caption,
                'policy_violations': [],
                'moderation_status': status,
                'human_review_required': review_needed,
                'confidence_score': 0.9 if self.nude_detector else 0.5
            }
            
            logger.info(f"Analysis complete: {status} (nudity: {nudity_score:.2f}%)")
            
            # Return as a simple object that can be converted to dict
            class Result:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                        
            return Result(**result_dict)
            
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting NudeNet Upload API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    api = NudeNetUploadAPI()
    api.run()