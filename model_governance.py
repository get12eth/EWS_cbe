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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                'accuracy': accuracy_score(actuals, predictions),
                'precision': precision_score(actuals, predictions, average='weighted', zero_division=0),
                'recall': recall_score(actuals, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(actuals, predictions, average='weighted', zero_division=0)
            }
            
            # Calculate AUC-ROC if probabilities are provided
            if probabilities and len(set(actuals)) == 2:  # Binary classification
                try:
                    metrics['auc_roc'] = roc_auc_score(actuals, probabilities)
                except:
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
                (evaluation_date, model_version, accuracy, precision, recall, f1_score, 
                 auc_roc, total_predictions, correct_predictions, true_positives, 
                 false_positives, true_negatives, false_negatives)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                evaluation_date.date(),
                self._get_model_version(),
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('auc_roc', 0),
                metrics.get('total_predictions', 0),
                metrics.get('correct_predictions', 0),
                metrics.get('true_positives_NPL', 0),  # Assuming NPL is the positive class
                metrics.get('false_positives_NPL', 0),
                metrics.get('true_negatives_NPL', 0),
                metrics.get('false_negatives_NPL', 0)
            ))
            
            conn.commit()
            logger.info(f"Model performance logged for {evaluation_date.date()}")
            
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
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
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Extract base value and SHAP values
            if hasattr(shap_values, 'base_values'):
                if isinstance(shap_values.base_values, np.ndarray):
                    base_value = float(shap_values.base_values[0])
                else:
                    base_value = float(shap_values.base_values[0][0])
            else:
                base_value = 0.0
            
            # Get SHAP values for the predicted class
            if isinstance(shap_values, list):
                # Multi-class case
                shap_vals = shap_values[prediction][0]
            else:
                # Binary case
                shap_vals = shap_values[0]
            
            # Create feature explanations
            feature_explanations = []
            for i, feature in enumerate(self.feature_cols):
                feature_value = float(X_processed[0][i]) if len(X_processed[0]) > i else 0.0
                shap_value = float(shap_vals[i]) if i < len(shap_vals) else 0.0
                
                feature_explanations.append({
                    'feature': feature,
                    'feature_value': feature_value,
                    'shap_value': shap_value,
                    'impact': 'positive' if shap_value > 0 else 'negative'
                })
            
            #Sort by absolute SHAP value
            feature_explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            #Get NPL probability if available
            npl_probability = 0.0
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_processed)
                if 'NPL' in list(self.label_encoder.classes_):
                    npl_idx = list(self.label_encoder.classes_).index('NPL')
                    npl_probability = float(probabilities[0][npl_idx])
            
            explanation = {
                'contract_code': customer_data.get('CONTRACT_CODE'),
                'predicted_status': predicted_status,
                'npl_probability': npl_probability,
                'base_value': base_value,
                'top_features': feature_explanations[:10],  # Top 10 features
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
        """Prepare customer data for prediction"""
        # Create DataFrame with required features
        data = {}
        for feature in self.feature_cols:
            data[feature] = customer_data.get(feature, 0)
        
        return pd.DataFrame([data])
    
    def _preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess data using the same pipeline as training"""
        try:
            data = df.copy()
            
            # Apply target encoder
            for col in self.target_encoder.mapping:
                if col in data.columns:
                    transformed = self.target_encoder.transform(data[[col]])
                    data[col] = transformed[col]
            
            # Encode remaining categorical columns
            for col in self.remaining_cats:
                if col in data.columns:
                    data[col] = pd.Categorical(data[col]).codes
            
            # Ensure all features exist
            for feature in self.feature_cols:
                if feature not in data.columns:
                    data[feature] = 0
            
            # Scale features
            X = data[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            # Return raw data as fallback
            return df[self.feature_cols].fillna(0).values
    
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
    
    def detect_data_drift(self) -> Dict:
        """Detect data drift between training and current data"""
        try:
            # Get current data statistics
            current_stats = self._get_current_data_stats()
            
            if not current_stats:
                return {'error': 'No current data available'}
            
            # Get training data statistics (stored or calculated)
            training_stats = self._get_training_data_stats()
            
            drift_results = {}
            
            for feature in self.feature_cols:
                if feature in current_stats and feature in training_stats:
                    current_mean = current_stats[feature]['mean']
                    current_std = current_stats[feature]['std']
                    training_mean = training_stats[feature]['mean']
                    training_std = training_stats[feature]['std']
                    
                    # Calculate drift score (simple statistical distance)
                    if training_std > 0 and current_std > 0:
                        drift_score = abs(current_mean - training_mean) / training_std
                    else:
                        drift_score = 0.0
                    
                    # Determine if drift is detected
                    is_drift = drift_score > 2.0  # 2 standard deviations threshold
                    
                    # Determine severity
                    if drift_score > 3.0:
                        severity = 'high'
                    elif drift_score > 2.0:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    drift_results[feature] = {
                        'drift_score': drift_score,
                        'is_drift_detected': is_drift,
                        'severity': severity,
                        'current_mean': current_mean,
                        'training_mean': training_mean,
                        'current_std': current_std,
                        'training_std': training_std
                    }
            
            # Log drift detection results
            self._log_data_drift(drift_results)
            
            return {
                'drift_results': drift_results,
                'features_with_drift': [f for f, r in drift_results.items() if r['is_drift_detected']],
                'detection_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to detect data drift: {e}")
            return {'error': str(e)}
    
    def _get_current_data_stats(self) -> Optional[Dict]:
        """Get statistics of current data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get recent data statistics
            cursor.execute(f"""
                SELECT {', '.join([f'AVG({f}) as {f}_mean, STDDEV({f}) as {f}_std' for f in self.feature_cols])}
                FROM customers 
                WHERE BUSINESS_DATE >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                AND PREDICTED_STATUS IS NOT NULL
            """)
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                return None
            
            stats = {}
            for feature in self.feature_cols:
                stats[feature] = {
                    'mean': result.get(f'{feature}_mean', 0),
                    'std': result.get(f'{feature}_std', 0)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get current data stats: {e}")
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def _get_training_data_stats(self) -> Dict:
        """Get training data statistics (would be stored during model training)"""
        # For now, return default values
        # In production, these should be stored during model training
        stats = {}
        for feature in self.feature_cols:
            stats[feature] = {
                'mean': 0.0,
                'std': 1.0
            }
        return stats
    
    def _log_data_drift(self, drift_results: Dict):
        """Log data drift detection results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for feature, result in drift_results.items():
                sql = """
                    INSERT INTO data_drift 
                    (detection_date, feature_name, training_distribution, current_distribution,
                     drift_score, is_drift_detected, severity, recommendation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                training_dist = {
                    'mean': result['training_mean'],
                    'std': result['training_std']
                }
                current_dist = {
                    'mean': result['current_mean'],
                    'std': result['current_std']
                }
                
                recommendation = self._generate_drift_recommendation(feature, result)
                
                cursor.execute(sql, (
                    datetime.now().date(),
                    feature,
                    json.dumps(training_dist),
                    json.dumps(current_dist),
                    result['drift_score'],
                    result['is_drift_detected'],
                    result['severity'],
                    recommendation
                ))
            
            conn.commit()
            logger.info("Data drift results logged")
            
        except Exception as e:
            logger.error(f"Failed to log data drift: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
    def _generate_drift_recommendation(self, feature: str, drift_result: Dict) -> str:
        """Generate recommendation for detected drift"""
        if not drift_result['is_drift_detected']:
            return "No action needed"
        
        severity = drift_result['severity']
        
        if severity == 'high':
            return f"High drift detected in {feature}. Consider retraining the model with recent data."
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
                    AVG(precision) as avg_precision,
                    AVG(recall) as avg_recall,
                    AVG(f1_score) as avg_f1_score,
                    AVG(auc_roc) as avg_auc_roc,
                    COUNT(*) as evaluation_count
                FROM model_performance 
                WHERE evaluation_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            performance_summary = cursor.fetchone()
            
            return {
                'recent_performance': recent_performance,
                'feature_importance': feature_importance,
                'recent_drift': recent_drift,
                'performance_summary': performance_summary,
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
