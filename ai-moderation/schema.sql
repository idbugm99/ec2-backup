-- Content Moderation Database Schema
-- Add this to your existing MuseNest database

CREATE TABLE IF NOT EXISTS content_moderation (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_path VARCHAR(255) NOT NULL,
    model_id INT NOT NULL,
    context_type ENUM('profile_pic', 'public_gallery', 'premium_gallery', 'private_content') NOT NULL,
    
    -- NudeNet Results
    nudity_score DECIMAL(5,2) DEFAULT 0.00,
    detected_parts JSON, -- {"breast": 95.5, "buttocks": 23.2, "genitalia": 2.1}
    
    -- Pose Analysis
    pose_classification VARCHAR(100),
    explicit_pose_score DECIMAL(5,2) DEFAULT 0.00,
    pose_keypoints JSON,
    
    -- BLIP Analysis  
    generated_caption TEXT,
    policy_violations JSON, -- ["explicit_pose", "sexual_act"]
    
    -- Final Decision
    moderation_status ENUM('approved', 'flagged', 'rejected') NOT NULL,
    human_review_required BOOLEAN DEFAULT FALSE,
    confidence_score DECIMAL(4,2) DEFAULT 0.00,
    reviewed_by INT NULL,
    reviewed_at TIMESTAMP NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL,
    
    -- Indexes for performance
    INDEX idx_model_context (model_id, context_type),
    INDEX idx_status (moderation_status),
    INDEX idx_review_required (human_review_required),
    INDEX idx_created_at (created_at)
);

-- Table to store moderation queue for human review
CREATE TABLE IF NOT EXISTS moderation_queue (
    id INT PRIMARY KEY AUTO_INCREMENT,
    content_moderation_id INT NOT NULL,
    priority ENUM('low', 'medium', 'high', 'urgent') DEFAULT 'medium',
    assigned_to INT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (content_moderation_id) REFERENCES content_moderation(id) ON DELETE CASCADE,
    FOREIGN KEY (assigned_to) REFERENCES users(id) ON DELETE SET NULL,
    
    INDEX idx_priority (priority),
    INDEX idx_assigned (assigned_to),  
    INDEX idx_created (created_at)
);

-- Table to track moderation rules and their changes
CREATE TABLE IF NOT EXISTS moderation_rules (
    id INT PRIMARY KEY AUTO_INCREMENT,
    context_type ENUM('profile_pic', 'public_gallery', 'premium_gallery', 'private_content') NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    rule_value JSON NOT NULL, -- Store rule configuration as JSON
    is_active BOOLEAN DEFAULT TRUE,
    created_by INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE,
    
    UNIQUE KEY unique_context_rule (context_type, rule_name),
    INDEX idx_context (context_type),
    INDEX idx_active (is_active)
);