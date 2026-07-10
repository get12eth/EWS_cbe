-- Comprehensive Database Schema for CBE Loan Risk Management System
-- Database: lon-default

SET @OLD_FOREIGN_KEY_CHECKS = @@FOREIGN_KEY_CHECKS;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS notification_logs;
DROP TABLE IF EXISTS case_activities;
DROP TABLE IF EXISTS cases;
DROP TABLE IF EXISTS shap_explanations;
DROP TABLE IF EXISTS feature_importance;
DROP TABLE IF EXISTS model_fairness;
DROP TABLE IF EXISTS model_performance;
DROP TABLE IF EXISTS data_drift;
DROP TABLE IF EXISTS alerts;
DROP TABLE IF EXISTS alert_rules;
DROP TABLE IF EXISTS prediction_results;
DROP TABLE IF EXISTS data_validation_logs;
DROP TABLE IF EXISTS anomaly_detection;
DROP TABLE IF EXISTS etl_pipeline_runs;
DROP TABLE IF EXISTS simulation_sector_results;
DROP TABLE IF EXISTS simulation_scenarios;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS system_config;
DROP TABLE IF EXISTS activity_logs;
DROP TABLE IF EXISTS ews_admins;
DROP TABLE IF EXISTS loan_table;
DROP TABLE IF EXISTS macro_indicators;
DROP TABLE IF EXISTS model_evaluations;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS customers;

SET FOREIGN_KEY_CHECKS = @OLD_FOREIGN_KEY_CHECKS;

-- 1. Enhanced Customer Table (Core Data)
CREATE TABLE IF NOT EXISTS customers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    CONTRACT_CODE VARCHAR(50) UNIQUE NOT NULL,
    DISTRICTNAME VARCHAR(100),
    CBE_REGION VARCHAR(100),
    BRANCHNAME VARCHAR(100),
    APPROVED_AMOUNT DECIMAL(15,2),
    GRANT_DATE DATE,
    EXPIRY_DATE DATE,
    TENURE VARCHAR(50),
    TERM VARCHAR(50),
    LOAN_TYPE VARCHAR(100),
    LTYPE VARCHAR(100),
    LOAN_DESCRIPTION TEXT,
    LOAN_PRODUCT VARCHAR(100),
    BUSINESS_DATE DATE,
    PRINCIPAL_OS DECIMAL(15,2),
    INTEREST_OS DECIMAL(15,2),
    PRINCIPAL_ARREARS DECIMAL(15,2),
    CURRENT_COMMITTMENT DECIMAL(15,2),
    INSTALLMENT_AMOUNT DECIMAL(15,2),
    ECONOMIC_SECTOR VARCHAR(100),
    INDUSTRY VARCHAR(100),
    OWNERSHIP VARCHAR(100),
    SECTOR VARCHAR(100),
    TERM_OF_PAYMENT VARCHAR(50),
    PRODUCT_OWNER VARCHAR(100),
    COLLATERAL_VALUE DECIMAL(15,2),
   
    -- Audit fields
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CREATED_BY VARCHAR(50),
    UPDATED_BY VARCHAR(50)
);

-- 1.5. Prediction Results Table
-- Stores model prediction results separate from customer data
CREATE TABLE IF NOT EXISTS prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    contract_code VARCHAR(50),
    
    -- Prediction results
    predicted_status VARCHAR(20), -- PAS, SME, SET, NPL
    npl_probability DECIMAL(8,6),
    pas_probability DECIMAL(8,6),
    sme_probability DECIMAL(8,6),
    set_probability DECIMAL(8,6),
    risk_level VARCHAR(20), -- Low Risk, Medium Risk, High Risk
    
    -- Metadata
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) DEFAULT 'v1.0',
    feature_count INT DEFAULT 31,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key relationships
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (contract_code) REFERENCES customers(CONTRACT_CODE),
    
    -- Indexes for performance
    INDEX idx_contract_code (contract_code),
    INDEX idx_prediction_date (prediction_date),
    INDEX idx_predicted_status (predicted_status),
    INDEX idx_customer_id (customer_id)
);

-- 2. ETL Engine Tables
-- Data validation logs
CREATE TABLE IF NOT EXISTS data_validation_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    batch_id VARCHAR(50),
    contract_code VARCHAR(50),
    validation_type VARCHAR(50), -- 'missing_coords', 'invalid_dates', 'amount_anomaly'
    severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    original_value TEXT,
    suggested_value TEXT,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'reviewed', 'fixed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_by VARCHAR(50),
    reviewed_at TIMESTAMP NULL,
    
    FOREIGN KEY (contract_code) REFERENCES customers(CONTRACT_CODE)
);

-- Anomaly detection results
CREATE TABLE IF NOT EXISTS anomaly_detection (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_date DATE,
    branch_name VARCHAR(100),
    metric_type VARCHAR(50), -- 'total_approved', 'avg_loan_amount', 'default_rate'
    current_value DECIMAL(15,2),
    baseline_value DECIMAL(15,2),
    deviation_percentage DECIMAL(8,2),
    anomaly_score DECIMAL(8,2),
    is_anomaly BOOLEAN DEFAULT FALSE,
    description TEXT,
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'investigated', 'resolved'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ETL pipeline runs
CREATE TABLE IF NOT EXISTS etl_pipeline_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20), -- 'running', 'completed', 'failed'
    records_processed INT,
    records_validated INT,
    records_failed INT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Strategy & Alerts Engine Tables
-- Alert rules configuration
CREATE TABLE IF NOT EXISTS alert_rules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    description TEXT,
    condition_type VARCHAR(50), -- 'npl_probability', 'loan_status_change', 'amount_anomaly'
    threshold_value DECIMAL(10,6),
    operator VARCHAR(10), -- '>', '<', '>=', '<=', '='
    severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    notification_channels JSON, -- ['email', 'sms', 'dashboard']
    recipients JSON, -- [{"role": "branch_manager", "branches": ["branch1", "branch2"]}, {"user_id": 123}]
    is_active BOOLEAN DEFAULT TRUE,
    escalation_hours INT DEFAULT 48,
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Alert instances
CREATE TABLE IF NOT EXISTS alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    alert_id VARCHAR(50) UNIQUE,
    rule_id INT,
    contract_code VARCHAR(50),
    branch_name VARCHAR(100),
    severity VARCHAR(20),
    title VARCHAR(200),
    description TEXT,
    current_value DECIMAL(15,2),
    threshold_value DECIMAL(15,2),
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'acknowledged', 'escalated', 'resolved'
    assigned_to VARCHAR(50), -- user_id or role
    due_date TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    resolution_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP NULL,
    acknowledged_by VARCHAR(50),
    escalated_at TIMESTAMP NULL,
    escalated_to VARCHAR(50),
    
    FOREIGN KEY (rule_id) REFERENCES alert_rules(id),
    FOREIGN KEY (contract_code) REFERENCES customers(CONTRACT_CODE)
);

-- Notification logs
CREATE TABLE IF NOT EXISTS notification_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    alert_id VARCHAR(50),
    channel VARCHAR(50), -- 'email', 'sms', 'dashboard'
    recipient VARCHAR(200), -- email address, phone number, or user_id
    subject VARCHAR(200),
    message TEXT,
    status VARCHAR(20), -- 'sent', 'failed', 'pending'
    sent_at TIMESTAMP,
    error_message TEXT,
    
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
);

-- 4. Case Management & Workflow Tables
-- Cases for high-risk loans
CREATE TABLE IF NOT EXISTS cases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_id VARCHAR(50) UNIQUE,
    contract_code VARCHAR(50),
    case_type VARCHAR(50), -- 'recovery', 'restructuring', 'write_off', 'investigation'
    priority VARCHAR(20), -- 'low', 'medium', 'high', 'urgent'
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'in_progress', 'resolved', 'closed'
    assigned_to VARCHAR(50), -- recovery officer ID
    assigned_by VARCHAR(50),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP,
    expected_resolution DATE,
    
    -- Case details
    risk_score DECIMAL(8,6),
    total_exposure DECIMAL(15,2),
    days_past_due INT,
    last_payment_date DATE,
    
    -- Resolution
    resolution_type VARCHAR(50), -- 'restructured', 'recovered', 'written_off', 'legal_action'
    resolution_amount DECIMAL(15,2),
    resolution_date DATE,
    resolution_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (contract_code) REFERENCES customers(CONTRACT_CODE)
);

-- Case activities and notes
CREATE TABLE IF NOT EXISTS case_activities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_id VARCHAR(50),
    activity_type VARCHAR(50), -- 'call', 'visit', 'email', 'note', 'payment_received', 'legal_action'
    description TEXT,
    outcome TEXT,
    next_action TEXT,
    next_action_date DATE,
    amount DECIMAL(15,2), -- if payment involved
    
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- 5. Model Governance & Explainability Tables
-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    evaluation_date DATE,
    model_version VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    avg_precision DECIMAL(5,4),
    npl_auc DECIMAL(5,4),
    total_predictions INT,
    correct_predictions INT,

    -- Confusion matrix
    true_positives INT,
    false_positives INT,
    true_negatives INT,
    false_negatives INT,

    -- Extended governance metrics
    confusion_matrix JSON,
    per_class_metrics JSON,
    npl_drift_score DECIMAL(8,6),
    prediction_drift_score DECIMAL(8,6),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature importance tracking
CREATE TABLE IF NOT EXISTS feature_importance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    evaluation_date DATE,
    model_version VARCHAR(50),
    feature_name VARCHAR(100),
    importance_score DECIMAL(10,8),
    importance_rank INT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SHAP explanations for individual predictions
CREATE TABLE IF NOT EXISTS shap_explanations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    contract_code VARCHAR(50),
    prediction_date TIMESTAMP,
    model_version VARCHAR(50),
    
    -- Overall prediction info
    predicted_status VARCHAR(20),
    npl_probability DECIMAL(8,6),
    base_value DECIMAL(8,6),
    
    -- Top contributing features (JSON array)
    top_features JSON, -- [{"feature": "LOAN_AGE_DAYS", "shap_value": 0.234, "feature_value": 180}]
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (contract_code) REFERENCES customers(CONTRACT_CODE)
);

-- Data drift detection
CREATE TABLE IF NOT EXISTS data_drift (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_date DATE,
    feature_name VARCHAR(100),
    training_distribution JSON, -- {"mean": 1000, "std": 500, "histogram": [...]}
    current_distribution JSON, -- {"mean": 1200, "std": 600, "histogram": [...]}
    drift_score DECIMAL(8,6),
    is_drift_detected BOOLEAN DEFAULT FALSE,
    severity VARCHAR(20), -- 'low', 'medium', 'high'
    recommendation TEXT,
    ks_statistic DECIMAL(8,6),
    p_value DECIMAL(12,10),
    current_count INT,
    reference_count INT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance ROC / Precision-Recall curves
CREATE TABLE IF NOT EXISTS model_performance_curves (
    id INT AUTO_INCREMENT PRIMARY KEY,
    evaluation_date DATE,
    model_version VARCHAR(50),
    curve_type VARCHAR(30),   -- 'roc' or 'pr'
    class_label VARCHAR(20),
    data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_eval (evaluation_date)
);

-- Explanation feedback (XAI refinement loop)
CREATE TABLE IF NOT EXISTS explanation_feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    contract_code VARCHAR(50),
    prediction_date TIMESTAMP NULL,
    helpful BOOLEAN,
    rating TINYINT NULL,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_contract (contract_code)
);

-- Automated model actions (retrain / investigate)
CREATE TABLE IF NOT EXISTS model_actions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    action_type VARCHAR(50),
    trigger_reason TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    detail JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Drift reference baseline snapshot
CREATE TABLE IF NOT EXISTS drift_reference (
    id INT AUTO_INCREMENT PRIMARY KEY,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sample_size INT,
    feature_cols JSON,
    reference_features JSON,
    reference_predictions JSON
);

-- Model fairness / bias tracking
CREATE TABLE IF NOT EXISTS model_fairness (
    id INT AUTO_INCREMENT PRIMARY KEY,
    evaluation_date TIMESTAMP,
    model_version VARCHAR(50),
    sensitive_attribute VARCHAR(100),
    group_value VARCHAR(200),
    selection_rate DECIMAL(8,6), -- proportion predicted positive (e.g., NPL)
    true_positive_rate DECIMAL(8,6),
    false_negative_rate DECIMAL(8,6),
    disparate_impact DECIMAL(8,6),
    metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit trail for critical actions (cases, alerts, governance changes)
CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(100), -- 'case_activity', 'case_assignment', 'alert_escalation', 'model_update'
    object_type VARCHAR(50), -- 'case', 'alert', 'model'
    object_id VARCHAR(100), -- case_id or alert_id or model_version
    performed_by VARCHAR(100),
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. What-If Simulation Tables
-- Simulation scenarios
CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(50) UNIQUE,
    scenario_name VARCHAR(100),
    description TEXT,
    created_by VARCHAR(50),
    
    -- Economic parameters
    inflation_rate DECIMAL(5,4),
    interest_rate_change DECIMAL(5,4),
    gdp_growth DECIMAL(5,4),
    unemployment_rate DECIMAL(5,4),
    
    -- Portfolio adjustments
    sector_exclusions JSON, -- ["Construction", "Real Estate"]
    sector_increments JSON, -- [{"sector": "Agriculture", "increase_percentage": 20}]
    
    -- Results
    total_portfolio_value DECIMAL(15,2),
    predicted_npl_count INT,
    predicted_npl_percentage DECIMAL(5,4),
    risk_adjusted_return DECIMAL(8,6),
    
    status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'running', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL
);

-- Simulation results by sector
CREATE TABLE IF NOT EXISTS simulation_sector_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(50),
    economic_sector VARCHAR(100),
    
    -- Baseline (current) metrics
    baseline_loan_count INT,
    baseline_portfolio_value DECIMAL(15,2),
    baseline_npl_count INT,
    baseline_npl_rate DECIMAL(5,4),
    
    -- Simulated metrics
    simulated_loan_count INT,
    simulated_portfolio_value DECIMAL(15,2),
    simulated_npl_count INT,
    simulated_npl_rate DECIMAL(5,4),
    
    -- Impact
    npl_change INT,
    npl_rate_change DECIMAL(5,4),
    portfolio_value_change DECIMAL(15,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (scenario_id) REFERENCES simulation_scenarios(scenario_id)
);

-- 7. User Management (Enhanced)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(200),
    full_name VARCHAR(200),
    role VARCHAR(50), -- 'admin', 'risk_officer', 'branch_manager', 'recovery_officer', 'data_analyst', 'executive'
    branch VARCHAR(100), -- for branch-level users
    department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);


-- 8. System Configuration
CREATE TABLE IF NOT EXISTS system_config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE,
    config_value TEXT,
    config_type VARCHAR(50), -- 'string', 'number', 'boolean', 'json'
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Indexes for performance
SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'customers'
                  AND index_name = 'idx_customers_contract_code');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_customers_contract_code ON customers(CONTRACT_CODE)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'customers'
                  AND index_name = 'idx_customers_branch');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_customers_branch ON customers(BRANCHNAME)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'customers'
                  AND index_name = 'idx_customers_business_date');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_customers_business_date ON customers(BUSINESS_DATE)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'alerts'
                  AND index_name = 'idx_alerts_contract_code');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_alerts_contract_code ON alerts(CONTRACT_CODE)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'alerts'
                  AND index_name = 'idx_alerts_status');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_alerts_status ON alerts(status)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'cases'
                  AND index_name = 'idx_cases_contract_code');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_cases_contract_code ON cases(CONTRACT_CODE)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'cases'
                  AND index_name = 'idx_cases_assigned_to');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_cases_assigned_to ON cases(assigned_to)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @exists = (SELECT COUNT(*) FROM information_schema.statistics
                WHERE table_schema = DATABASE() AND table_name = 'cases'
                  AND index_name = 'idx_cases_status');
SET @sql = IF(@exists = 0,
              'CREATE INDEX idx_cases_status ON cases(status)',
              'SELECT 1');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Insert default admin, DAO, and risk officer users
INSERT INTO users (username, password_hash, email, full_name, role) 
VALUES
('admin', '$pbkdf2-sha256$29000$855TKsXYu/c.h1AKAWDs/Q$hcgwnfCuojqSUOkDo1Xr/5IM4mTYAEYB4ZkL.uzk1ks', 'admin@cbe.com.et', 'System Administrator', 'admin'),
('dao', '$pbkdf2-sha256$29000$5by3NqYUYixFKAUgBOBcKw$/o2N63L7GHrmLcEWhdOgwuu7jK8kQYm5iBRpsltzP1k', 'dao@cbe.com.et', 'District Administrator Officer', 'dao'),
('risk_officer', '$pbkdf2-sha256$29000$GWNMKeX8/z/nfC9FiHFOCQ$Mx97l6uufK1mymeb5tGkVhZtAcxrCEMXh/6u/GkdWEg', 'riskofficer@cbe.com.et', 'Risk Officer', 'risk_officer')
ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash);

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, config_type, description) VALUES
('model_version', '1.0', 'string', 'Current model version'),
('default_npl_threshold', '0.4', 'number', 'Default NPL probability threshold for alerts'),
('data_retention_days', '1095', 'number', 'Days to retain historical data'),
('max_file_size_mb', '100', 'number', 'Maximum file size for uploads in MB'),
('enable_email_notifications', 'true', 'boolean', 'Enable email notifications'),
('enable_sms_notifications', 'false', 'boolean', 'Enable SMS notifications')
ON DUPLICATE KEY UPDATE config_value = VALUES(config_value);
