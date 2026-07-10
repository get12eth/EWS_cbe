"""
Model Governance & Explainability (XAI) Module for CBE Loan Risk Management System
Handles model performance tracking, SHAP explanations, and drift detection
"""

import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from scipy.stats import ks_2samp, chi2_contingency
import shap
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Human-readable metadata for model features used in explanations.
FEATURE_DISPLAY_NAMES = {
    'PRINCIPAL_OS': ('Principal Outstanding', 'Remaining principal balance on the loan'),
    'INTEREST_OS': ('Interest Outstanding', 'Unpaid interest accrued'),
    'PRINCIPAL_ARREARS': ('Principal Arrears', 'Overdue principal amount'),
    'INTEREST_ARREARS': ('Interest Arrears', 'Overdue interest amount'),
    'CURRENT_COMMITTMENT': ('Current Commitment', 'Total current exposure'),
    'APPROVED_AMOUNT': ('Approved Amount', 'Original sanctioned loan amount'),
    'INSTALLMENT_AMOUNT': ('Installment Amount', 'Scheduled periodic installment'),
    'COLLATERAL_VALUE': ('Collateral Value', 'Value of pledged collateral'),
    'INTEREST_RATE': ('Interest Rate', 'Contractual interest rate'),
    'LOAN_AGE_DAYS': ('Loan Age (days)', 'Days since the loan was disbursed'),
    'TOTAL_LOAN_DAYS': ('Total Loan Tenure (days)', 'Full contractual loan duration'),
    'PRINCIPAL_OS_RATIO': ('Principal OS Ratio', 'Outstanding principal / approved amount'),
    'ARREARS_RATIO': ('Arrears Ratio', 'Total arrears / principal outstanding'),
    'ECONOMIC_SECTOR': ('Economic Sector', 'Sector the borrower operates in'),
    'OWNERSHIP': ('Ownership Type', 'Legal ownership structure of the borrower'),
    'BRANCHNAME': ('Branch', 'Originating CBE branch'),
    'LOAN_TYPE': ('Loan Type', 'Product category of the loan'),
    'LTYPE': ('Loan Sub-type', 'Detailed loan product classification'),
    'TERM': ('Term', 'Repayment term description'),
    'TENURE': ('Tenure', 'Loan tenure label'),
    'RISK_GRADE': ('Risk Grade', 'Internal risk rating'),
}

# Drift detection thresholds (Kolmogorov-Smirnov D statistic)
DRIFT_D_MEDIUM = 0.10
DRIFT_D_HIGH = 0.20
DRIFT_P_VALUE = 0.05
DRIFT_REFERENCE_SAMPLE = 1000

class ModelGovernance:
    def __init__(self, db_config: Dict, model_path: str = "models/loan_prediction_assets_1.pkl"):
        self.db_config = db_config
        self.conn = None
        self.model_path = model_path
        self.model_assets = None
        self.explainer = None
        self.load_model()
        
    def get_connection(self):
        """Get database connection"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.conn = mysql.connector.connect(**self.db_config)
            return self.conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_model(self):
        """Load the trained model and assets"""
        try:
            self.model_assets = joblib.load(self.model_path)
            self.model = self.model_assets['model']
            self.feature_cols = self.model_assets['features']
            self.label_encoder = self.model_assets['label_encoder']
            self.scaler = self.model_assets['scaler']
            self.target_encoder = self.model_assets['target_encoder']
            self.remaining_cats = self.model_assets.get('remaining_cats', [])
            
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Ensure governance tables/columns exist (idempotent)
        try:
            self.ensure_schema()
        except Exception as e:
            logger.warning(f"ensure_schema skipped/failed: {e}")

    def ensure_schema(self):
        """Create governance tables and add missing columns (idempotent)."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_curves (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    evaluation_date DATE,
                    model_version VARCHAR(50),
                    curve_type VARCHAR(30),
                    class_label VARCHAR(20),
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_eval (evaluation_date)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS explanation_feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    contract_code VARCHAR(50),
                    prediction_date TIMESTAMP NULL,
                    helpful BOOLEAN,
                    rating TINYINT NULL,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_contract (contract_code)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_actions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    action_type VARCHAR(50),
                    trigger_reason TEXT,
                    status VARCHAR(20) DEFAULT 'pending',
                    detail JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_reference (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sample_size INT,
                    feature_cols JSON,
                    reference_features JSON,
                    reference_predictions JSON
                )
            """)

            for table, column, ctype in [
                ('model_performance', 'avg_precision', 'DECIMAL(5,4)'),
                ('model_performance', 'npl_auc', 'DECIMAL(5,4)'),
                ('model_performance', 'confusion_matrix', 'JSON'),
                ('model_performance', 'per_class_metrics', 'JSON'),
                ('model_performance', 'npl_drift_score', 'DECIMAL(8,6)'),
                ('model_performance', 'prediction_drift_score', 'DECIMAL(8,6)'),
                ('data_drift', 'ks_statistic', 'DECIMAL(8,6)'),
                ('data_drift', 'p_value', 'DECIMAL(12,10)'),
                ('data_drift', 'current_count', 'INT'),
                ('data_drift', 'reference_count', 'INT'),
                ('customers', 'LOAN_STATUS', 'VARCHAR(50)'),
            ]:
                self._add_column_if_missing(cursor, table, column, ctype)

            conn.commit()
        except Exception as e:
            logger.error(f"ensure_schema failed: {e}")
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def _add_column_if_missing(self, cursor, table, column, col_type):
        """Add a column to a table only if it does not already exist."""
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s
            """, (table, column))
            row = cursor.fetchone()
            if not row or int(row[0]) == 0:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                logger.info(f"Added column {table}.{column}")
        except Exception as e:
            logger.warning(f"Could not add column {table}.{column}: {e}")

    def _human_feature(self, feature: str):
        """Return a human-readable name and description for a feature."""
        if feature in FEATURE_DISPLAY_NAMES:
            return FEATURE_DISPLAY_NAMES[feature]
        name = feature.replace('_', ' ').title()
        return name, ''

    @staticmethod
    def _jload(value):
        """Safely parse a JSON column (mysql-connector may already decode it)."""
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return None

    def initialize_explainer(self):
        """Initialize SHAP explainer"""
        try:
            # Try different explainer types based on model
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based model
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except:
                    self.explainer = shap.Explainer(self.model)
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    np.zeros((1, len(self.feature_cols)))
                )
            
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def calculate_model_performance(self, predictions: List, actuals: List, probabilities: List = None) -> Dict:
        """Calculate model performance metrics"""
        try:
            metrics = {
                'accuracy': float(accuracy_score(actuals, predictions)),
                'precision': float(precision_score(actuals, predictions, average='weighted', zero_division=0)),
                'recall': float(recall_score(actuals, predictions, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(actuals, predictions, average='weighted', zero_division=0))
            }

            # Calculate AUC-ROC if probabilities are provided
            if probabilities and len(set(actuals)) == 2:  # Binary classification
                try:
                    metrics['auc_roc'] = float(roc_auc_score(actuals, probabilities))
                except Exception:
                    metrics['auc_roc'] = 0.0
            else:
                metrics['auc_roc'] = 0.0
            
            # Calculate confusion matrix components
            unique_labels = list(set(actuals + predictions))
            for label in unique_labels:
                tp = sum(1 for a, p in zip(actuals, predictions) if a == label and p == label)
                fp = sum(1 for a, p in zip(actuals, predictions) if a != label and p == label)
                tn = sum(1 for a, p in zip(actuals, predictions) if a != label and p != label)
                fn = sum(1 for a, p in zip(actuals, predictions) if a == label and p != label)
                
                metrics[f'true_positives_{label}'] = tp
                metrics[f'false_positives_{label}'] = fp
                metrics[f'true_negatives_{label}'] = tn
                metrics[f'false_negatives_{label}'] = fn
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate model performance: {e}")
            return {}
    
    def log_model_performance(self, evaluation_date: datetime, metrics: Dict):
        """Log model performance metrics to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO model_performance 
                (evaluation_date, model_version, accuracy, precision_score, recall, f1_score, 
                 auc_roc, avg_precision, npl_auc, total_predictions, correct_predictions,
                 true_positives, false_positives, true_negatives, false_negatives,
                 confusion_matrix, per_class_metrics, npl_drift_score, prediction_drift_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                evaluation_date.date(),
                self._get_model_version(),
                float(metrics.get('accuracy', 0)),
                float(metrics.get('precision', 0)),
                float(metrics.get('recall', 0)),
                float(metrics.get('f1_score', 0)),
                float(metrics.get('auc_roc', 0)),
                float(metrics.get('avg_precision', 0)),
                float(metrics.get('npl_auc', 0)),
                int(metrics.get('total_predictions', 0)),
                int(metrics.get('correct_predictions', 0)),
                int(metrics.get('true_positives_NPL', 0)),  # Assuming NPL is the positive class
                int(metrics.get('false_positives_NPL', 0)),
                int(metrics.get('true_negatives_NPL', 0)),
                int(metrics.get('false_negatives_NPL', 0)),
                json.dumps(metrics.get('confusion_matrix')) if metrics.get('confusion_matrix') else None,
                json.dumps(metrics.get('per_class_metrics')) if metrics.get('per_class_metrics') else None,
                float(metrics.get('npl_drift_score', 0)) if metrics.get('npl_drift_score') is not None else None,
                float(metrics.get('prediction_drift_score', 0)) if metrics.get('prediction_drift_score') is not None else None
            ))
            
            conn.commit()
            logger.info(f"Model performance logged for {evaluation_date.date()}")
            
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
    def _downsample_curve(self, curve: Dict, max_points: int = 120) -> Dict:
        """Reduce the number of points in a curve dict for compact storage."""
        try:
            keys = list(curve.keys())
            n = len(curve[keys[0]])
            if n <= max_points:
                return curve
            idx = np.linspace(0, n - 1, max_points).astype(int)
            return {k: [v[i] for i in idx] for k, v in curve.items()}
        except Exception:
            return curve

    def _compute_curve_metrics(self, actuals, predictions, probability_vectors, classes, positive_label='NPL') -> Dict:
        """Compute confusion matrix, per-class metrics, and ROC/PR curves (OvR)."""
        result = {'confusion_matrix': None, 'per_class_metrics': None,
                  'roc': None, 'pr': None, 'avg_precision': 0.0, 'npl_auc': 0.0, 'error': None}
        try:
            labels = sorted(set(actuals) | set(predictions))
            cm = confusion_matrix(actuals, predictions, labels=labels)
            result['confusion_matrix'] = {'labels': labels, 'matrix': cm.tolist()}

            rep = classification_report(actuals, predictions, labels=labels, output_dict=True, zero_division=0)
            result['per_class_metrics'] = {lab: rep[lab] for lab in labels if lab in rep}
        except Exception as e:
            result['error'] = str(e)
            logger.warning(f"Confusion/per-class metrics failed: {e}")

        if probability_vectors and positive_label in classes:
            idx = classes.index(positive_label)
            y_true = [1 if a == positive_label else 0 for a in actuals]
            y_score = [vec[idx] for vec in probability_vectors]
            try:
                fpr, tpr, thr = roc_curve(y_true, y_score)
                result['roc'] = self._downsample_curve({
                    'fpr': [float(x) for x in fpr],
                    'tpr': [float(x) for x in tpr],
                    'thresholds': [float(x) for x in thr]
                })
                result['npl_auc'] = float(roc_auc_score(y_true, y_score))
            except Exception as e:
                logger.warning(f"ROC computation failed: {e}")
            try:
                prec, rec, pthr = precision_recall_curve(y_true, y_score)
                result['pr'] = self._downsample_curve({
                    'precision': [float(x) for x in prec],
                    'recall': [float(x) for x in rec],
                    'thresholds': [float(x) for x in pthr]
                })
                result['avg_precision'] = float(average_precision_score(y_true, y_score))
            except Exception as e:
                logger.warning(f"PR computation failed: {e}")
        return result

    def _log_performance_curves(self, eval_date, curves: Dict):
        """Persist ROC/PR curve points for the latest evaluation."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            version = self._get_model_version()
            for curve_type, key in [('roc', 'roc'), ('pr', 'pr')]:
                data = curves.get(key)
                if data:
                    cursor.execute("""
                        INSERT INTO model_performance_curves
                        (evaluation_date, model_version, curve_type, class_label, data)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (eval_date.date(), version, curve_type, 'NPL', json.dumps(data)))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log performance curves: {e}")
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def evaluate_model_performance(self, as_of_date: datetime = None) -> Dict:
        """Evaluate the latest model predictions against actual loan outcomes.

        Joins each customer's most recent prediction (prediction_results) with the
        actual loan status stored on the customers table (LOAN_STATUS), computes
        classification metrics via calculate_model_performance, ROC/PR curves, and
        persists them through log_model_performance so the governance dashboard is
        populated. Also runs drift detection and triggers investigation if needed.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT
                    p.predicted_status,
                    p.npl_probability,
                    p.pas_probability,
                    p.sme_probability,
                    p.set_probability,
                    c.LOAN_STATUS AS actual_status
                FROM prediction_results p
                JOIN (
                    SELECT contract_code, MAX(id) AS max_id
                    FROM prediction_results
                    GROUP BY contract_code
                ) latest
                    ON p.contract_code = latest.contract_code AND p.id = latest.max_id
                JOIN customers c
                    ON c.CONTRACT_CODE = p.contract_code
                WHERE c.LOAN_STATUS IS NOT NULL
                  AND c.LOAN_STATUS <> ''
                  AND p.predicted_status IS NOT NULL
            """)
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                # Diagnose WHY: count predictions vs customers with a stored actual outcome.
                try:
                    cur2 = conn.cursor()
                    cur2.execute("SELECT COUNT(*) FROM prediction_results")
                    total_preds = cur2.fetchone()[0]
                    cur2.execute("SELECT COUNT(*) FROM customers WHERE LOAN_STATUS IS NOT NULL AND LOAN_STATUS <> ''")
                    actuals = cur2.fetchone()[0]
                    cur2.close()
                except Exception:
                    total_preds, actuals = None, None
                reason = (
                    f"No prediction/actual pairs: {total_preds} predictions exist but only "
                    f"{actuals} customers have an actual LOAN_STATUS. Accuracy/F1/ROC require "
                    f"actual outcomes — load them via ETL (source loan book) or set "
                    f"customers.LOAN_STATUS from the system of record, then re-evaluate."
                )
                logger.warning(f"Model evaluation skipped: {reason}")
                return {'evaluated': False, 'reason': reason, 'evaluation_count': 0,
                        'total_predictions': total_preds, 'customers_with_actuals': actuals}

            predictions = [str(r['predicted_status']).strip().upper() for r in rows]
            actuals = [str(r['actual_status']).strip().upper() for r in rows]

            # Build per-class probability vectors aligned with the label encoder classes
            probability_vectors = None
            try:
                classes = list(self.label_encoder.classes_)
                prob_col = {cls: f"{str(cls).lower()}_probability" for cls in classes}
                probability_vectors = [
                    [float(r.get(prob_col[cls]) or 0.0) for cls in classes]
                    for r in rows
                ]
            except Exception as e:
                logger.warning(f"Could not build probability vectors for AUC: {e}")

            # Calculate base metrics (AUC handled separately below)
            metrics = self.calculate_model_performance(predictions, actuals, None)
            if not metrics:
                return {'evaluated': False, 'reason': 'compute_failed', 'evaluation_count': len(rows)}

            # Compute AUC properly (binary or multiclass one-vs-rest)
            if probability_vectors and len(set(actuals)) >= 2:
                try:
                    y_true = self.label_encoder.transform(actuals)
                    y_score = np.array(probability_vectors)
                    if y_score.shape[1] == len(classes):
                        metrics['auc_roc'] = float(
                            roc_auc_score(y_true, y_score, multi_class='ovr', labels=range(len(classes)))
                        )
                except Exception as e:
                    logger.warning(f"Multiclass AUC computation failed: {e}")

            # Compute confusion matrix, per-class metrics and ROC/PR curves
            curve_metrics = self._compute_curve_metrics(actuals, predictions, probability_vectors, classes)
            metrics['confusion_matrix'] = curve_metrics['confusion_matrix']
            metrics['per_class_metrics'] = curve_metrics['per_class_metrics']
            metrics['avg_precision'] = curve_metrics['avg_precision']
            if curve_metrics.get('npl_auc'):
                metrics['npl_auc'] = curve_metrics['npl_auc']

            correct = sum(1 for a, p in zip(actuals, predictions) if a == p)
            metrics['total_predictions'] = len(rows)
            metrics['correct_predictions'] = correct

            eval_date = as_of_date or datetime.now()
            self.log_model_performance(eval_date, metrics)
            self._log_performance_curves(eval_date, curve_metrics)

            # Run drift detection + automated investigation as part of the monitor loop
            drift = self.detect_data_drift()
            investigation = None
            if isinstance(drift, dict) and drift.get('drift_results'):
                high = [f for f, r in drift['drift_results'].items() if r.get('severity') == 'high']
                if high:
                    investigation = self.trigger_investigation({
                        'type': 'drift',
                        'features': high,
                        'npl_drift_score': drift.get('npl_drift_score'),
                        'prediction_drift_score': drift.get('prediction_drift_score')
                    })
                metrics['npl_drift_score'] = drift.get('npl_drift_score')
                metrics['prediction_drift_score'] = drift.get('prediction_drift_score')

            metrics['evaluated'] = True
            metrics['evaluation_count'] = len(rows)
            metrics['evaluation_date'] = eval_date.isoformat()
            if investigation:
                metrics['investigation'] = investigation
            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")
            return {'evaluated': False, 'reason': str(e), 'evaluation_count': 0}

    def calculate_feature_importance(self) -> Dict:
        """Calculate feature importance using model's built-in method and SHAP"""
        try:
            importance_data = {}
            
            # Get model's built-in feature importance
            if hasattr(self.model, 'feature_importances_'):
                model_importance = self.model.feature_importances_
                importance_data['model_importance'] = dict(zip(self.feature_cols, model_importance))
            
            # Calculate SHAP importance if explainer is available
            if not self.explainer:
                self.initialize_explainer()
            
            if self.explainer:
                # Get sample data for SHAP calculation
                sample_data = self._get_sample_data()
                if sample_data is not None:
                    shap_values = self.explainer.shap_values(sample_data)
                    
                    # Calculate mean absolute SHAP values for global importance
                    if isinstance(shap_values, list):
                        # Multi-class case
                        shap_importance = np.mean(np.abs(shap_values), axis=0)
                    else:
                        # Binary case
                        shap_importance = np.mean(np.abs(shap_values), axis=0)
                    
                    # Average across samples if needed
                    if len(shap_importance.shape) > 1:
                        shap_importance = np.mean(shap_importance, axis=0)
                    
                    importance_data['shap_importance'] = dict(zip(self.feature_cols, shap_importance))
            
            return importance_data
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def log_feature_importance(self, evaluation_date: datetime):
        """Log feature importance to database"""
        try:
            importance_data = self.calculate_feature_importance()
            
            if not importance_data:
                return
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Use SHAP importance if available, otherwise use model importance
            importance_scores = importance_data.get('shap_importance', importance_data.get('model_importance', {}))
            
            # Sort by importance and rank
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            sql = """
                INSERT INTO feature_importance 
                (evaluation_date, model_version, feature_name, importance_score, importance_rank)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            for rank, (feature, score) in enumerate(sorted_features, 1):
                cursor.execute(sql, (
                    evaluation_date.date(),
                    self._get_model_version(),
                    feature,
                    score,
                    rank
                ))
            
            conn.commit()
            logger.info(f"Feature importance logged for {evaluation_date.date()}")
            
        except Exception as e:
            logger.error(f"Failed to log feature importance: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def generate_shap_explanation(self, customer_data: Dict) -> Dict:
        """Generate SHAP explanation for individual prediction"""
        try:
            if not self.explainer:
                self.initialize_explainer()
            
            if not self.explainer:
                return {'error': 'SHAP explainer not available'}
            
            # Prepare data for prediction
            X_df = self._prepare_customer_data(customer_data)
            X_processed = self._preprocess_data(X_df)
            
            # Get prediction
            prediction = self.model.predict(X_processed)[0]
            predicted_status = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get SHAP values for all classes
            sv = self.explainer.shap_values(X_processed)
            classes = list(self.label_encoder.classes_)

            # Normalise SHAP output into a list of per-class arrays (sample 0)
            if isinstance(sv, list):
                class_shap = [np.asarray(s[0]) for s in sv]
                if hasattr(sv[0], 'base_values'):
                    base_values = [float(np.asarray(s.base_values).reshape(-1)[0]) for s in sv]
                else:
                    base_values = [0.0] * len(sv)
            else:
                class_shap = [np.asarray(sv[0])]
                if hasattr(sv, 'base_values'):
                    base_values = [float(np.asarray(sv.base_values).reshape(-1)[0])]
                else:
                    base_values = [0.0]

            # Per-class predicted probabilities
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(X_processed)
                except Exception:
                    probabilities = None

            def prob_for(cls):
                if probabilities is not None and cls in classes:
                    return float(probabilities[0][classes.index(cls)])
                return 0.0

            npl_probability = prob_for('NPL')

            # Build a per-class contribution breakdown (waterfall-ready)
            per_class = []
            for ci, cls in enumerate(classes):
                contribs = []
                for i, feature in enumerate(self.feature_cols):
                    try:
                        fv = float(X_processed[0][i])
                    except Exception:
                        fv = 0.0
                    try:
                        sv_val = float(class_shap[ci][i])
                    except Exception:
                        sv_val = 0.0
                    name, desc = self._human_feature(feature)
                    contribs.append({
                        'feature': feature,
                        'display_name': name,
                        'description': desc,
                        'feature_value': fv,
                        'shap_value': sv_val,
                    })
                contribs.sort(key=lambda x: abs(x['shap_value']), reverse=True)
                per_class.append({
                    'class': cls,
                    'probability': prob_for(cls),
                    'base_value': base_values[ci] if ci < len(base_values) else 0.0,
                    'contributions': contribs[:15],
                })

            # Order classes by probability (predicted class first)
            per_class.sort(key=lambda x: (x['class'] != predicted_status, -x['probability']))

            predicted_block = next((b for b in per_class if b['class'] == predicted_status), per_class[0])
            feature_explanations = [{
                'feature': c['feature'],
                'display_name': c['display_name'],
                'description': c['description'],
                'feature_value': c['feature_value'],
                'shap_value': c['shap_value'],
                'impact': 'positive' if c['shap_value'] > 0 else 'negative'
            } for c in predicted_block['contributions']]

            explanation = {
                'contract_code': customer_data.get('CONTRACT_CODE'),
                'predicted_status': predicted_status,
                'npl_probability': npl_probability,
                'class_probabilities': {b['class']: b['probability'] for b in per_class},
                'base_value': predicted_block['base_value'],
                'per_class': per_class,
                'top_features': feature_explanations[:10],
                'model_version': self._get_model_version(),
                'explanation_date': datetime.now().isoformat()
            }
            
            # Log to database
            self._log_shap_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}")
            return {'error': str(e)}
    
    def _prepare_customer_data(self, customer_data: Dict) -> pd.DataFrame:
        """Prepare customer data for prediction (reconstructs engineered date features)."""
        data = {}
        for feature in self.feature_cols:
            data[feature] = customer_data.get(feature, 0)

        # Reconstruct engineered features when raw dates are available, so the
        # explanation matches what the model actually predicted.
        try:
            gd = customer_data.get('GRANT_DATE')
            ed = customer_data.get('EXPIRY_DATE')
            bd = customer_data.get('BUSINESS_DATE')
            if 'TOTAL_LOAN_DAYS' in self.feature_cols and gd and ed:
                try:
                    data['TOTAL_LOAN_DAYS'] = (pd.to_datetime(ed) - pd.to_datetime(gd)).days
                except Exception:
                    pass
            if 'LOAN_AGE_DAYS' in self.feature_cols and gd and bd:
                try:
                    data['LOAN_AGE_DAYS'] = (pd.to_datetime(bd) - pd.to_datetime(gd)).days
                except Exception:
                    pass
        except Exception:
            pass

        return pd.DataFrame([data])
    
    def _preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess data into the numeric model input (robust to categoricals).

        Mirrors the prediction pipeline in main.prepare_model_input: numeric columns
        are coerced, categorical columns are deterministically encoded, and all
        columns are guaranteed numeric before scaling.
        """
        try:
            data = df.copy()

            # Ensure all training features exist (missing -> 0)
            for feature in self.feature_cols:
                if feature not in data.columns:
                    data[feature] = 0

            X = data[self.feature_cols].copy()

            numeric_cols, categorical_cols = [], []
            for col in X.columns:
                coerced = pd.to_numeric(X[col], errors='coerce')
                if coerced.notna().any() and coerced.notna().all():
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

            # Numeric columns: coerce, fill missing with column mean (then 0)
            for col in numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                mean = X[col].mean()
                X[col] = X[col].fillna(mean if pd.notna(mean) else 0)

            # Categorical columns: prefer the trained target encoder, else stable hash
            mapping = getattr(self.target_encoder, 'mapping', None) if self.target_encoder else None
            if mapping:
                for col in [c for c in categorical_cols if c in mapping]:
                    try:
                        X[col] = self.target_encoder.transform(X[[col]])[col]
                        categorical_cols.remove(col)
                    except Exception as e:
                        logger.warning(f"Target encoding failed for {col}: {e}")

            for col in categorical_cols:
                X[col] = X[col].astype(str).fillna('Unknown')
                X[col] = X[col].map(lambda v: float(abs(hash(v)) % 1000))

            # Final safety: force every column numeric (strings -> 0)
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            if self.scaler is not None:
                try:
                    return self.scaler.transform(X)
                except Exception as e:
                    logger.warning(f"Scaler transform failed in SHAP preprocessing: {e}")
            return X.values.astype(np.float64)

        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            # Safe fallback: zero matrix of the correct shape
            return np.zeros((len(df), len(self.feature_cols)), dtype=np.float64)
    
    def _get_sample_data(self, n_samples: int = 100) -> Optional[np.ndarray]:
        """Get sample data for SHAP calculation"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(f"""
                SELECT {', '.join(self.feature_cols)}
                FROM customers 
                WHERE PREDICTED_STATUS IS NOT NULL
                ORDER BY RAND()
                LIMIT %s
            """, (n_samples,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            return self._preprocess_data(df)
            
        except Exception as e:
            logger.error(f"Failed to get sample data: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def _log_shap_explanation(self, explanation: Dict):
        """Log SHAP explanation to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO shap_explanations 
                (contract_code, prediction_date, model_version, predicted_status,
                 npl_probability, base_value, top_features)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                explanation['contract_code'],
                datetime.now(),
                explanation['model_version'],
                explanation['predicted_status'],
                explanation['npl_probability'],
                explanation['base_value'],
                json.dumps(explanation['top_features'])
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log SHAP explanation: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
    def _get_customer_feature_columns(self):
        """Return the subset of model features that exist as columns in customers."""
        if getattr(self, '_customer_feature_cols', None) is not None:
            return self._customer_feature_cols
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COLUMN_NAME FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = 'customers'
            """)
            cols = {row[0] for row in cursor.fetchall()}
            cursor.close()
            self._customer_feature_cols = [f for f in self.feature_cols if f in cols]
        except Exception as e:
            logger.warning(f"Could not introspect customer columns: {e}")
            self._customer_feature_cols = list(self.feature_cols)
        return self._customer_feature_cols

    def _sample_feature_data(self, n: int = DRIFT_REFERENCE_SAMPLE):
        """Sample feature vectors + prediction outputs from current customers."""
        try:
            feat_cols = self._get_customer_feature_columns()
            if not feat_cols:
                return None
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cols_sql = ', '.join(f"`{c}`" for c in feat_cols)
            cursor.execute(f"""
                SELECT {cols_sql}, pr.predicted_status, pr.npl_probability
                FROM customers c
                JOIN (
                    SELECT contract_code, MAX(id) AS max_id FROM prediction_results
                    GROUP BY contract_code
                ) latest ON c.CONTRACT_CODE = latest.contract_code
                JOIN prediction_results pr
                    ON pr.contract_code = latest.contract_code AND pr.id = latest.max_id
                WHERE c.BUSINESS_DATE >= DATE_SUB(NOW(), INTERVAL 90 DAY)
                  AND pr.predicted_status IS NOT NULL
                ORDER BY RAND()
                LIMIT %s
            """, (n,))
            rows = cursor.fetchall()
            cursor.close()
            if not rows:
                return None
            features, predicted, npl_prob = [], [], []
            for r in rows:
                vec = []
                for c in feat_cols:
                    v = r.get(c)
                    try:
                        vec.append(float(v))
                    except (TypeError, ValueError):
                        vec.append(0.0)
                features.append(vec)
                predicted.append(str(r.get('predicted_status') or '').upper())
                try:
                    npl_prob.append(float(r.get('npl_probability') or 0.0))
                except (TypeError, ValueError):
                    npl_prob.append(0.0)
            return {
                'feature_cols': feat_cols,
                'features': features,
                'predicted': predicted,
                'npl_prob': npl_prob,
                'sample_size': len(rows)
            }
        except Exception as e:
            logger.error(f"Failed to sample feature data: {e}")
            return None

    def _get_reference_sample(self):
        """Load the latest captured drift reference baseline."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT feature_cols, reference_features, reference_predictions, sample_size
                FROM drift_reference ORDER BY captured_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            preds = self._jload(row['reference_predictions']) if row['reference_predictions'] else {}
            return {
                'feature_cols': self._jload(row['feature_cols']) if row['feature_cols'] else [],
                'features': self._jload(row['reference_features']) if row['reference_features'] else [],
                'predicted': preds.get('predicted', []),
                'npl_prob': preds.get('npl_prob', []),
                'sample_size': row['sample_size']
            }
        except Exception as e:
            logger.error(f"Failed to get reference sample: {e}")
            return None

    def capture_drift_reference(self, n: int = DRIFT_REFERENCE_SAMPLE):
        """Snap the current population as the drift reference baseline."""
        sample = self._sample_feature_data(n)
        if not sample:
            return {'captured': False, 'reason': 'no_data'}
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drift_reference
                (sample_size, feature_cols, reference_features, reference_predictions)
                VALUES (%s, %s, %s, %s)
            """, (
                sample['sample_size'],
                json.dumps(sample['feature_cols']),
                json.dumps(sample['features']),
                json.dumps({'predicted': sample['predicted'], 'npl_prob': sample['npl_prob']})
            ))
            conn.commit()
            cursor.close()
            return {'captured': True, 'sample_size': sample['sample_size']}
        except Exception as e:
            logger.error(f"Failed to capture drift reference: {e}")
            return {'captured': False, 'reason': str(e)}

    def detect_data_drift(self, reference: Dict = None) -> Dict:
        """Detect input + prediction drift via the Kolmogorov-Smirnov two-sample test."""
        try:
            if reference is None:
                reference = self._get_reference_sample()
            if not reference or not reference.get('features'):
                cap = self.capture_drift_reference()
                return {
                    'baseline_captured': True,
                    'message': 'No reference baseline found; captured current data as baseline.',
                    'capture': cap,
                    'detection_date': datetime.now().isoformat()
                }

            current = self._sample_feature_data(len(reference.get('features', [])) or DRIFT_REFERENCE_SAMPLE)
            if not current:
                return {'error': 'No current data available for drift detection'}

            feat_cols = reference.get('feature_cols') or current['feature_cols']
            drift_results = {}
            for i, feat in enumerate(feat_cols):
                try:
                    ref_vec = np.array([row[i] for row in reference['features']], dtype=float)
                    cur_vec = np.array([row[i] for row in current['features']], dtype=float)
                except Exception:
                    continue
                if len(ref_vec) < 5 or len(cur_vec) < 5:
                    continue
                try:
                    stat, pval = ks_2samp(cur_vec, ref_vec)
                except Exception as e:
                    logger.warning(f"KS failed for {feat}: {e}")
                    continue

                if stat >= DRIFT_D_HIGH:
                    severity = 'high'
                elif stat >= DRIFT_D_MEDIUM:
                    severity = 'medium'
                else:
                    severity = 'low'
                is_drift = (stat >= DRIFT_D_MEDIUM) and (pval < DRIFT_P_VALUE)

                drift_results[feat] = {
                    'drift_score': float(stat),
                    'ks_statistic': float(stat),
                    'p_value': float(pval),
                    'is_drift_detected': bool(is_drift),
                    'severity': severity,
                    'current_mean': float(np.mean(cur_vec)),
                    'reference_mean': float(np.mean(ref_vec)),
                    'current_std': float(np.std(cur_vec)),
                    'reference_std': float(np.std(ref_vec)),
                    'current_count': int(len(cur_vec)),
                    'reference_count': int(len(ref_vec))
                }

            # Prediction-output drift: NPL probability distribution shift
            npl_drift = 0.0
            try:
                ref_p = np.array(reference.get('npl_prob') or [], dtype=float)
                cur_p = np.array(current['npl_prob'], dtype=float)
                if len(ref_p) >= 5 and len(cur_p) >= 5:
                    stat, _ = ks_2samp(cur_p, ref_p)
                    npl_drift = float(stat)
            except Exception as e:
                logger.warning(f"NPL probability drift failed: {e}")

            # Prediction-label drift: categorical output distribution shift (TVD)
            prediction_drift = 0.0
            try:
                ref_pred = reference.get('predicted') or []
                cur_pred = current['predicted']
                if ref_pred and cur_pred:
                    ref_counts = pd.Series(ref_pred).value_counts()
                    cur_counts = pd.Series(cur_pred).value_counts()
                    labels = sorted(set(ref_counts.index) | set(cur_counts.index))
                    ref_dist = np.array([ref_counts.get(l, 0) for l in labels], dtype=float)
                    cur_dist = np.array([cur_counts.get(l, 0) for l in labels], dtype=float)
                    if ref_dist.sum() > 0 and cur_dist.sum() > 0:
                        ref_dist = ref_dist / ref_dist.sum()
                        cur_dist = cur_dist / cur_dist.sum()
                        prediction_drift = float(0.5 * np.sum(np.abs(cur_dist - ref_dist)))
            except Exception as e:
                logger.warning(f"Prediction label drift failed: {e}")

            self._log_data_drift(drift_results, npl_drift, prediction_drift)

            return {
                'drift_results': drift_results,
                'features_with_drift': [f for f, r in drift_results.items() if r['is_drift_detected']],
                'npl_drift_score': npl_drift,
                'prediction_drift_score': prediction_drift,
                'detection_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to detect data drift: {e}")
            return {'error': str(e)}

    def _log_data_drift(self, drift_results: Dict, npl_drift: float = 0.0, prediction_drift: float = 0.0):
        """Log data drift detection results (input + prediction-output drift)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            for feature, result in drift_results.items():
                recommendation = self._generate_drift_recommendation(feature, result)
                cursor.execute("""
                    INSERT INTO data_drift
                    (detection_date, feature_name, training_distribution, current_distribution,
                     drift_score, is_drift_detected, severity, recommendation,
                     ks_statistic, p_value, current_count, reference_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now().date(),
                    feature,
                    json.dumps({'mean': result.get('reference_mean'), 'std': result.get('reference_std')}),
                    json.dumps({'mean': result.get('current_mean'), 'std': result.get('current_std')}),
                    result['drift_score'],
                    result['is_drift_detected'],
                    result['severity'],
                    recommendation,
                    result.get('ks_statistic'),
                    result.get('p_value'),
                    result.get('current_count'),
                    result.get('reference_count')
                ))

            # Log aggregate prediction-output drift as a synthetic feature row
            combined = max(npl_drift, prediction_drift)
            if combined > 0:
                sev = 'high' if combined >= DRIFT_D_HIGH else ('medium' if combined >= DRIFT_D_MEDIUM else 'low')
                cursor.execute("""
                    INSERT INTO data_drift
                    (detection_date, feature_name, training_distribution, current_distribution,
                     drift_score, is_drift_detected, severity, recommendation,
                     ks_statistic, p_value, current_count, reference_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now().date(),
                    'PREDICTION_OUTPUT',
                    json.dumps({'npl_drift': npl_drift, 'prediction_drift': prediction_drift}),
                    json.dumps({}),
                    combined,
                    bool(combined >= DRIFT_D_MEDIUM),
                    sev,
                    'Prediction-output drift detected. Investigate model staleness and consider retraining.',
                    npl_drift,
                    None,
                    None,
                    None
                ))

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log data drift: {e}")
            conn.rollback()
        finally:
            cursor.close()

    def _generate_drift_recommendation(self, feature: str, drift_result: Dict) -> str:
        """Generate recommendation for detected drift"""
        if not drift_result.get('is_drift_detected'):
            return "No action needed"
        severity = drift_result.get('severity')
        if severity == 'high':
            return f"High drift detected in {feature}. Automated investigation and model retraining recommended."
        elif severity == 'medium':
            return f"Medium drift detected in {feature}. Monitor closely and plan model retraining."
        else:
            return f"Low drift detected in {feature}. Continue monitoring."
    
    def _get_model_version(self) -> str:
        """Get current model version"""
        # This could be stored in a config file or database
        return "1.0"
    
    def get_model_governance_dashboard(self) -> Dict:
        """Get model governance data for dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get recent model performance
            cursor.execute("""
                SELECT * FROM model_performance 
                WHERE evaluation_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                ORDER BY evaluation_date DESC
                LIMIT 10
            """)
            
            recent_performance = cursor.fetchall()
            
            # Get latest feature importance
            cursor.execute("""
                SELECT feature_name, importance_score, importance_rank
                FROM feature_importance 
                WHERE evaluation_date = (
                    SELECT MAX(evaluation_date) FROM feature_importance
                )
                ORDER BY importance_rank
                LIMIT 15
            """)
            
            feature_importance = cursor.fetchall()
            
            # Get recent drift detection
            cursor.execute("""
                SELECT feature_name, drift_score, is_drift_detected, severity
                FROM data_drift 
                WHERE detection_date >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                ORDER BY drift_score DESC
                LIMIT 10
            """)
            
            recent_drift = cursor.fetchall()
            
            # Get performance summary
            cursor.execute("""
                SELECT 
                    AVG(accuracy) as avg_accuracy,
                    AVG(precision_score) as avg_precision,
                    AVG(recall) as avg_recall,
                    AVG(f1_score) as avg_f1_score,
                    AVG(auc_roc) as avg_auc_roc,
                    COUNT(*) as evaluation_count
                FROM model_performance 
                WHERE evaluation_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            performance_summary = cursor.fetchone()

            # Latest ROC/PR curves for visualisation
            cursor.execute("""
                SELECT curve_type, class_label, data
                FROM model_performance_curves
                WHERE evaluation_date = (
                    SELECT MAX(evaluation_date) FROM model_performance_curves
                )
            """)
            curves_rows = cursor.fetchall()
            latest_curves = {}
            for c in curves_rows:
                latest_curves[c['curve_type']] = {
                    'class_label': c['class_label'],
                    'data': self._jload(c['data'])
                }

            # Latest evaluation detail (confusion matrix + per-class metrics)
            cursor.execute("""
                SELECT evaluation_date, accuracy, f1_score, avg_precision, npl_auc,
                       confusion_matrix, per_class_metrics
                FROM model_performance
                ORDER BY evaluation_date DESC
                LIMIT 1
            """)
            latest_eval = cursor.fetchone()
            if latest_eval:
                latest_eval = {
                    'evaluation_date': latest_eval['evaluation_date'].isoformat() if hasattr(latest_eval['evaluation_date'], 'isoformat') else str(latest_eval['evaluation_date']),
                    'accuracy': latest_eval['accuracy'],
                    'f1_score': latest_eval['f1_score'],
                    'avg_precision': latest_eval['avg_precision'],
                    'npl_auc': latest_eval['npl_auc'],
                    'confusion_matrix': self._jload(latest_eval['confusion_matrix']),
                    'per_class_metrics': self._jload(latest_eval['per_class_metrics'])
                }

            # Pending automated investigations
            cursor.execute("""
                SELECT id, action_type, status, created_at, detail
                FROM model_actions
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            pending_actions = cursor.fetchall()

            return {
                'recent_performance': recent_performance,
                'feature_importance': feature_importance,
                'recent_drift': recent_drift,
                'performance_summary': performance_summary,
                'latest_curves': latest_curves,
                'latest_evaluation': latest_eval,
                'pending_actions': pending_actions,
                'feedback_summary': self.get_feedback_summary(),
                'model_version': self._get_model_version(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model governance dashboard: {e}")
            return {}
        finally:
            cursor.close()

    def compute_fairness_metrics(self, sensitive_attribute: str, positive_label: str = 'NPL') -> Dict:
        """Compute fairness metrics (selection rate, disparate impact, TPR) per group for a sensitive attribute.

        sensitive_attribute: column name in `customers` (e.g., 'OWNERSHIP', 'BRANCHNAME', 'ECONOMIC_SECTOR')
        positive_label: label considered as the positive class (e.g., 'NPL')
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)

            # Fetch latest prediction per customer joined to customer attribute
            cursor.execute("""
                SELECT c.%s AS group_value, pr.predicted_status, pr.prediction_date
                FROM prediction_results pr
                JOIN customers c ON pr.customer_id = c.id
                WHERE pr.prediction_date = (
                    SELECT MAX(pr2.prediction_date) FROM prediction_results pr2 WHERE pr2.customer_id = pr.customer_id
                )
            """ % (sensitive_attribute,))

            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return {'error': 'No prediction data available'}

            # Aggregate per group
            import collections
            group_stats = collections.defaultdict(lambda: {'count': 0, 'positive': 0})
            total = 0
            total_positive = 0

            for r in rows:
                grp = r.get('group_value') or 'UNKNOWN'
                total += 1
                group_stats[grp]['count'] += 1
                if str(r.get('predicted_status')).upper() == str(positive_label).upper():
                    group_stats[grp]['positive'] += 1
                    total_positive += 1

            overall_selection_rate = total_positive / total if total else 0.0

            results = []
            for grp, stats in group_stats.items():
                sel_rate = stats['positive'] / stats['count'] if stats['count'] else 0.0
                di = sel_rate / overall_selection_rate if overall_selection_rate > 0 else None

                results.append({
                    'group_value': grp,
                    'count': stats['count'],
                    'positive_count': stats['positive'],
                    'selection_rate': round(sel_rate, 6),
                    'disparate_impact': round(di, 6) if di is not None else None
                })

            # Log results to DB
            self._log_fairness_results(sensitive_attribute, results)

            return {
                'sensitive_attribute': sensitive_attribute,
                'overall_selection_rate': round(overall_selection_rate, 6),
                'groups': results,
                'evaluation_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to compute fairness metrics: {e}")
            return {'error': str(e)}

    def _log_fairness_results(self, sensitive_attribute: str, groups: List[Dict]):
        """Persist fairness metrics to model_fairness table"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            version = self._get_model_version()
            for g in groups:
                cursor.execute("""
                    INSERT INTO model_fairness
                    (evaluation_date, model_version, sensitive_attribute, group_value, selection_rate, disparate_impact, metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now(), version, sensitive_attribute, g['group_value'], g['selection_rate'], g.get('disparate_impact'), json.dumps(g)
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log fairness results: {e}")
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def get_feedback_summary(self) -> Dict:
        """Aggregate explanation feedback (helpful vs not) for the XAI interface."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) AS helpful,
                    SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END) AS not_helpful
                FROM explanation_feedback
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            row = cursor.fetchone()
            cursor.close()
            total = int(row['total'] or 0)
            helpful = int(row['helpful'] or 0)
            return {
                'total': total,
                'helpful': helpful,
                'not_helpful': int(row['not_helpful'] or 0),
                'helpful_rate': round(helpful / total, 4) if total else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get feedback summary: {e}")
            return {'total': 0, 'helpful': 0, 'not_helpful': 0, 'helpful_rate': 0.0}

    def log_explanation_feedback(self, contract_code: str, helpful: bool, comment: str = None,
                                 prediction_date: datetime = None) -> Dict:
        """Record whether an explanation was helpful to refine the XAI interface."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO explanation_feedback
                (contract_code, prediction_date, helpful, comment, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                contract_code,
                prediction_date or datetime.now(),
                bool(helpful),
                comment,
                datetime.now()
            ))
            conn.commit()
            cursor.close()
            return {'success': True}
        except Exception as e:
            logger.error(f"Failed to log explanation feedback: {e}")
            return {'success': False, 'error': str(e)}

    def get_explanation_for_contract(self, contract_code: str) -> Dict:
        """Return the latest stored SHAP explanation for a contract (with feedback state)."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM shap_explanations
                WHERE contract_code = %s
                ORDER BY prediction_date DESC
                LIMIT 1
            """, (contract_code,))
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return {'error': 'No explanation found for contract'}
            row['top_features'] = self._jload(row['top_features']) if row.get('top_features') else []
            return row
        except Exception as e:
            logger.error(f"Failed to get explanation: {e}")
            return {'error': str(e)}

    def set_alert_callback(self, callback):
        """Register a callback used to raise alerts when investigations are triggered."""
        self._alert_callback = callback

    def trigger_investigation(self, context: Dict) -> Dict:
        """Record an automated investigation/retraining action and raise an alert."""
        try:
            detail = {
                'trigger': context,
                'recommended_action': 'retrain_or_investigate',
                'created_at': datetime.now().isoformat()
            }
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_actions (action_type, trigger_reason, status, detail)
                VALUES (%s, %s, %s, %s)
            """, (
                context.get('type', 'investigate'),
                json.dumps(context),
                'pending',
                json.dumps(detail)
            ))
            action_id = cursor.lastrowid
            conn.commit()
            cursor.close()

            cb = getattr(self, '_alert_callback', None)
            if callable(cb):
                try:
                    cb('Model drift investigation triggered', context)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")

            self._audit_event('model_investigation', detail)
            return {'action_id': action_id, 'status': 'pending', 'detail': detail}
        except Exception as e:
            logger.error(f"Failed to trigger investigation: {e}")
            return {'error': str(e)}

    def _audit_event(self, event_type: str, payload: Dict):
        """Append a governance event to the audit trail."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (event_type, object_type, details, created_at)
                VALUES (%s, %s, %s, %s)
            """, (event_type, 'model', json.dumps(payload), datetime.now()))
            conn.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")

    def run_monitoring_cycle(self) -> Dict:
        """Full monitoring loop: evaluate performance, detect drift, trigger investigation."""
        result = {'performance': None, 'drift': None, 'investigation': None}
        try:
            result['performance'] = self.evaluate_model_performance()
            drift = self.detect_data_drift()
            result['drift'] = drift
            if isinstance(drift, dict) and drift.get('features_with_drift'):
                high = [f for f, r in drift.get('drift_results', {}).items() if r.get('severity') == 'high']
                if high:
                    result['investigation'] = self.trigger_investigation({
                        'type': 'drift',
                        'features': high,
                        'npl_drift_score': drift.get('npl_drift_score'),
                        'prediction_drift_score': drift.get('prediction_drift_score')
                    })
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            result['error'] = str(e)
        return result

# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Bant@6963',
        'database': 'lon-default'
    }
    
    # Initialize Model Governance
    model_gov = ModelGovernance(db_config)
    
    # Generate SHAP explanation for a sample customer
    sample_customer = {
        'CONTRACT_CODE': 'C000001',
        'APPROVED_AMOUNT': 500000,
        'PRINCIPAL_OS': 450000,
        'LOAN_AGE_DAYS': 180,
        'ECONOMIC_SECTOR': 'Agriculture'
    }
    
    # explanation = model_gov.generate_shap_explanation(sample_customer)
    # print(explanation)
