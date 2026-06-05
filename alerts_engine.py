"""
Strategy & Alerts Engine for CBE Loan Risk Management System
Handles alert rules, notifications, and escalation logic
"""

import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
import pandas as pd

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertsEngine:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        self.default_thresholds = {
            'npl_probability_high': 0.7,
            'npl_probability_medium': 0.5,
            'npl_probability_low': 0.3,
            'sme_prediction_threshold': 0.5,  # SME probability threshold (0-1)
            'days_past_due_critical': 90,
            'days_past_due_high': 60,
            'days_past_due_medium': 30,
            'amount_anomaly_threshold': 3.0
        }
        
    def get_connection(self):
        """Get database connection"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.conn = mysql.connector.connect(**self.db_config)
            return self.conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def create_alert_rule(self, rule_data: Dict) -> Dict:
        """Create a new alert rule"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO alert_rules 
                (rule_name, description, condition_type, threshold_value, operator,
                 severity, notification_channels, recipients, is_active, escalation_hours, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, 
            (
                rule_data['rule_name'],
                rule_data.get('description', ''),
                rule_data['condition_type'],
                rule_data['threshold_value'],
                rule_data['operator'],
                rule_data['severity'],
                json.dumps(rule_data.get('notification_channels', [])),
                json.dumps(rule_data.get('recipients', [])),
                rule_data.get('is_active', True),
                rule_data.get('escalation_hours', 48),
                rule_data.get('created_by', 'system')
            ))
            
            rule_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created alert rule: {rule_data['rule_name']} (ID: {rule_id})")
            return {'success': True, 'rule_id': rule_id}
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            conn.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def create_default_sme_alert_rules(self) -> Dict:
        """Create default SME prediction alert rules"""
        sme_alert_rules = [
            {
                'rule_name': 'SME Prediction Alert',
                'description': 'Alert when loan is predicted as SME status - requires officer attention',
                'condition_type': 'sme_prediction',
                'threshold_value': 0.5,
                'operator': '>=',
                'severity': 'medium',
                'notification_channels': ['email', 'dashboard'],
                'recipients': [{'role': 'branch_manager'}],
                'escalation_hours': 48,
                'created_by': 'system'
            },
            {
                'rule_name': 'High Priority SME Alert',
                'description': 'High priority alert for SME predictions requiring immediate attention',
                'condition_type': 'sme_prediction',
                'threshold_value': 0.75,
                'operator': '>=',
                'severity': 'high',
                'notification_channels': ['email', 'dashboard', 'sms'],
                'recipients': [{'role': 'branch_manager'}, {'role': 'regional_manager'}],
                'escalation_hours': 24,
                'created_by': 'system'
            }
        ]
        
        results = []
        for rule_data in sme_alert_rules:
            result = self.create_alert_rule(rule_data)
            results.append(result)
        
        return {
            'success': True,
            'created_rules': len([r for r in results if r.get('success')]),
            'results': results
        }
    
    def evaluate_alert_conditions(self, customer_data: Dict) -> List[Dict]:
        """Evaluate all active alert rules against customer data"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            #Get all active alert rules
            cursor.execute("SELECT * FROM alert_rules WHERE is_active = TRUE")
            rules = cursor.fetchall()
            
            triggered_alerts = []
            
            for rule in rules:
                alert = self._evaluate_rule(rule, customer_data)
                if alert:
                    triggered_alerts.append(alert)
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert conditions: {e}")
            return []
        finally:
            cursor.close()
    
    def _evaluate_rule(self, rule: Dict, customer_data: Dict) -> Optional[Dict]:
        """Evaluate a single alert rule"""
        condition_type = rule['condition_type']
        threshold = rule['threshold_value']
        operator = rule['operator']
        
        triggered = False
        current_value = None
        
        try:
            if condition_type == 'npl_probability':
                current_value = float(customer_data.get('NPL_PROBABILITY', 0))
                triggered = self._compare_values(current_value, threshold, operator)
                
            elif condition_type == 'sme_prediction':
                # SME prediction rule: support both predicted status and SME probability checks
                # Try to read SME probability from multiple possible keys
                sme_prob = None
                for key in ('SME_PROBABILITY', 'sme_probability', 'sme_prob', 'SME_PROB'):
                    if key in customer_data and customer_data.get(key) is not None:
                        try:
                            sme_prob = float(customer_data.get(key))
                            break
                        except Exception:
                            continue

                # If a numeric threshold is provided, compare using operator; otherwise fall back to status check
                if sme_prob is not None and isinstance(threshold, (int, float)):
                    current_value = sme_prob
                    triggered = self._compare_values(current_value, float(threshold), operator)
                else:
                    current_status = customer_data.get('PREDICTED_STATUS') or customer_data.get('predicted_status') or ''
                    target_status = 'SME'
                    triggered = str(current_status).upper() == target_status
                    current_value = 1 if triggered else 0
                
            elif condition_type == 'loan_status_change':
                current_status = customer_data.get('LOAN_STATUS', '')
                target_status = threshold
                triggered = current_status == target_status
                current_value = 1 if triggered else 0
                
            elif condition_type == 'days_past_due':
                # Calculate days past due from business date and last payment
                business_date = customer_data.get('BUSINESS_DATE')
                last_payment = customer_data.get('LAST_PAYMENT_DATE')
                
                if business_date and last_payment:
                    days_past_due = (pd.to_datetime(business_date) - pd.to_datetime(last_payment)).days
                    current_value = days_past_due
                    triggered = self._compare_values(days_past_due, threshold, operator)
                    
            elif condition_type == 'amount_anomaly':
                # This would require historical data comparison
                # For now, skip this complex evaluation
                pass
                
            elif condition_type == 'data_quality_score':
                current_value = float(customer_data.get('DATA_QUALITY_SCORE', 100))
                triggered = self._compare_values(current_value, threshold, operator)
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule['rule_name']}: {e}")
            return None
        
        if triggered:
            return {
                'rule_id': rule['id'],
                'contract_code': customer_data.get('CONTRACT_CODE'),
                'branch_name': customer_data.get('BRANCHNAME'),
                'severity': rule['severity'],
                'title': f"{rule['rule_name']} - {customer_data.get('CONTRACT_CODE')}",
                'description': self._generate_alert_description(rule, customer_data, current_value),
                'current_value': current_value,
                'threshold_value': threshold,
                'notification_channels': json.loads(rule['notification_channels']),
                'recipients': json.loads(rule['recipients'])
            }
        
        return None
    
    def _compare_values(self, current: float, threshold: float, operator: str) -> bool:
        """Compare values based on operator"""
        if operator == '>':
            return current > threshold
        elif operator == '>=':
            return current >= threshold
        elif operator == '<':
            return current < threshold
        elif operator == '<=':
            return current <= threshold
        elif operator == '=':
            return current == threshold
        return False
    
    def _generate_alert_description(self, rule: Dict, customer_data: Dict, current_value: float) -> str:
        """Generate alert description"""
        condition_type = rule['condition_type']
        threshold = rule['threshold_value']
        
        if condition_type == 'npl_probability':
            return f"Loan {customer_data.get('CONTRACT_CODE')} has NPL probability of {current_value:.4f}, exceeding threshold of {threshold}"
        elif condition_type == 'sme_prediction':
            return f"Loan {customer_data.get('CONTRACT_CODE')} predicted as SME - requires officer attention for monitoring"
        elif condition_type == 'loan_status_change':
            return f"Loan {customer_data.get('CONTRACT_CODE')} status changed to {customer_data.get('LOAN_STATUS')}"
        elif condition_type == 'days_past_due':
            return f"Loan {customer_data.get('CONTRACT_CODE')} is {current_value} days past due, exceeding threshold of {threshold} days"
        elif condition_type == 'data_quality_score':
            return f"Loan {customer_data.get('CONTRACT_CODE')} has data quality score of {current_value:.2f}, below threshold of {threshold}"
        else:
            return f"Alert triggered for loan {customer_data.get('CONTRACT_CODE')}"
    
    def create_alert(self, alert_data: Dict) -> Dict:
        """Create or update an alert (upsert pattern)
        
        If an alert already exists for this contract_code and rule_id (in open/acknowledged status),
        update it with new values. Otherwise, create a new alert.
        """
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            contract_code = alert_data.get('contract_code', '')
            rule_id = alert_data.get('rule_id', None)
            
            # Check if alert already exists for this contract and rule (in active status)
            existing_alert = None
            if contract_code and rule_id:
                cursor.execute("""
                    SELECT alert_id, id FROM alerts 
                    WHERE contract_code = %s AND rule_id = %s AND status IN ('open', 'acknowledged')
                    LIMIT 1
                """, (contract_code, rule_id))
                existing_alert = cursor.fetchone()
            
            if existing_alert:
                # UPDATE existing alert
                uid = existing_alert['alert_id']
                cursor.execute("""
                    UPDATE alerts 
                    SET severity = %s, title = %s, description = %s, 
                        current_value = %s, threshold_value = %s, 
                        branch_name = %s, created_at = NOW()
                    WHERE alert_id = %s
                """, (
                    alert_data.get('severity', 'medium'),
                    alert_data.get('title', alert_data.get('risk_signal', 'Alert')),
                    alert_data.get('description', ''),
                    alert_data.get('current_value', 0.0),
                    alert_data.get('threshold_value', None),
                    alert_data.get('branch_name', None),
                    uid
                ))
                conn.commit()
                logger.info(f"Updated existing alert: {uid} for contract {contract_code}")
                action = 'updated'
            else:
                # INSERT new alert
                uid = alert_data.get('alert_id') or str(uuid.uuid4())
                
                sql = """
                    INSERT INTO alerts 
                    (alert_id, rule_id, contract_code, branch_name, severity, title, description, current_value, threshold_value, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(sql, (
                    uid,
                    rule_id,
                    contract_code,
                    alert_data.get('branch_name', None),
                    alert_data.get('severity', 'medium'),
                    alert_data.get('title', alert_data.get('risk_signal', 'Alert')),
                    alert_data.get('description', ''),
                    alert_data.get('current_value', 0.0),
                    alert_data.get('threshold_value', None),
                    'open',
                    datetime.now()
                ))
                
                conn.commit()
                logger.info(f"Created new alert: {uid} for contract {contract_code}")
                action = 'created'
            
            # Send notifications using the varchar alert id
            self._send_notifications(uid, alert_data)
            
            return {'success': True, 'alert_id': uid, 'action': action}
            
        except Exception as e:
            logger.error(f"Failed to create/update alert: {e}")
            conn.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def _get_rule_escalation_hours(self, rule_id: int) -> int:
        """Get escalation hours for a rule"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT escalation_hours FROM alert_rules WHERE id = %s", (rule_id,))
            result = cursor.fetchone()
            return result[0] if result else 48
        except Exception:
            return 48
        finally:
            cursor.close()
    
    def _send_notifications(self, alert_id: str, alert_data: Dict):
        """Send notifications for an alert"""
        channels = alert_data.get('notification_channels', [])
        recipients = alert_data.get('recipients', [])
        
        for channel in channels:
            if channel == 'email':
                self._send_email_notifications(alert_id, alert_data, recipients)
            elif channel == 'sms':
                self._send_sms_notifications(alert_id, alert_data, recipients)
            elif channel == 'dashboard':
                # Dashboard notifications are handled by frontend polling
                pass
    
    def _send_email_notifications(self, alert_id: str, alert_data: Dict, recipients: List[Dict]):
        """Send email notifications"""
        try:
            #This is a placeholder - implement actual email sending logic
            #You would need to configure SMTP settings
            
            for recipient in recipients:
                if recipient.get('role') == 'branch_manager' and alert_data.get('branch_name'):
                    # Get branch manager email
                    email = self._get_branch_manager_email(alert_data['branch_name'])
                    if email:
                        self._send_email(email, alert_data['title'], alert_data['description'])
                        
                elif recipient.get('user_id'):
                    # Get user email
                    email = self._get_user_email(recipient['user_id'])
                    if email:
                        self._send_email(email, alert_data['title'], alert_data['description'])
            
            # Log notification
            self._log_notification(alert_id, 'email', 'recipients', alert_data['title'], 'sent')
            
        except Exception as e:
            logger.error(f"Failed to send email notifications: {e}")
            self._log_notification(alert_id, 'email', 'recipients', alert_data['title'], 'failed', str(e))
    
    def _send_email(self, to_email: str, subject: str, body: str):
        """Send individual email"""
        # Placeholder implementation
        # Configure SMTP settings in production
        logger.info(f"Sending email to {to_email}: {subject}")
        
        # Example SMTP configuration (replace with actual settings):
        # smtp_server = "smtp.gmail.com"
        # smtp_port = 587
        # smtp_username = "your_email@gmail.com"
        # smtp_password = "your_password"
        
        # msg = MIMEMultipart()
        # msg['From'] = smtp_username
        # msg['To'] = to_email
        # msg['Subject'] = subject
        # msg.attach(MIMEText(body, 'plain'))
        
        # server = smtplib.SMTP(smtp_server, smtp_port)
        # server.starttls()
        # server.login(smtp_username, smtp_password)
        # server.send_message(msg)
        # server.quit()
    
    def _send_sms_notifications(self, alert_id: str, alert_data: Dict, recipients: List[Dict]):
        """Send SMS notifications"""
        try:
            # Placeholder for SMS implementation
            # You would need to integrate with an SMS service provider
            
            for recipient in recipients:
                if recipient.get('role') == 'branch_manager' and alert_data.get('branch_name'):
                    phone = self._get_branch_manager_phone(alert_data['branch_name'])
                    if phone:
                        self._send_sms(phone, f"ALERT: {alert_data['title']} - {alert_data['description']}")
            
            self._log_notification(alert_id, 'sms', 'recipients', alert_data['title'], 'sent')
            
        except Exception as e:
            logger.error(f"Failed to send SMS notifications: {e}")
            self._log_notification(alert_id, 'sms', 'recipients', alert_data['title'], 'failed', str(e))
    
    def _send_sms(self, phone_number: str, message: str):
        """Send individual SMS"""
        # Placeholder implementation
        # Integrate with SMS provider like Twilio, Africa's Talking, etc.
        logger.info(f"Sending SMS to {phone_number}: {message}")
    
    def _get_branch_manager_email(self, branch_name: str) -> Optional[str]:
        """Get branch manager email from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT email FROM users 
                WHERE role = 'branch_manager' AND branch = %s AND is_active = TRUE
                LIMIT 1
            """, (branch_name,))
            
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception:
            return None
        finally:
            cursor.close()
    
    def _get_branch_manager_phone(self, branch_name: str) -> Optional[str]:
        """Get branch manager phone from database"""
        # This would require adding phone field to users table
        # For now, return None
        return None
    
    def _get_user_email(self, user_id: str) -> Optional[str]:
        """Get user email from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT email FROM users WHERE username = %s AND is_active = TRUE", (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception:
            return None
        finally:
            cursor.close()
    
    def _log_notification(self, alert_id: str, channel: str, recipient: str, subject: str, status: str, error_message: str = None):
        """Log notification attempt"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO notification_logs 
                (alert_id, channel, recipient, subject, message, status, sent_at, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                alert_id,
                channel,
                recipient,
                subject,
                subject,  # Using subject as message for now
                status,
                datetime.now(),
                error_message
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log notification: {e}")
        finally:
            cursor.close()
    
    def check_alert_escalations(self):
        """Check for alerts that need escalation"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get alerts that are past due and not resolved
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE status IN ('open', 'acknowledged') 
                AND due_date < NOW()
                AND escalated_at IS NULL
            """)
            
            alerts_to_escalate = cursor.fetchall()
            
            for alert in alerts_to_escalate:
                self._escalate_alert(alert)
            
            return len(alerts_to_escalate)
            
        except Exception as e:
            logger.error(f"Failed to check alert escalations: {e}")
            return 0
        finally:
            cursor.close()

    def populate_alerts_from_predictions(self, since_hours: int = 24) -> Dict:
        """Scan recent prediction results and create alerts based on active rules.

        since_hours: lookback window in hours to consider recent predictions
        Returns summary dict with counts
        """
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            # Load active rules that involve predictions
            cursor.execute("SELECT * FROM alert_rules WHERE is_active = TRUE")
            rules = cursor.fetchall()

            created = 0
            skipped = 0
            errors = []

            # Prepare time window
            cursor2 = conn.cursor(dictionary=True)
            for rule in rules:
                condition = rule['condition_type']
                threshold = rule['threshold_value']
                operator = rule['operator']

                # Build base query joining latest prediction results and customers
                sql = f"""
                    SELECT pr.*, c.CONTRACT_CODE, c.BRANCHNAME
                    FROM prediction_results pr
                    JOIN customers c ON pr.customer_id = c.id
                    WHERE pr.prediction_date >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                """

                params = (since_hours,)

                try:
                    cursor2.execute(sql, params)
                    preds = cursor2.fetchall()
                except Exception as e:
                    errors.append(str(e))
                    continue

                for p in preds:
                    try:
                        customer_data = dict(p)
                        # normalize keys for AlertsEngine evaluation
                        # pass both probability keys and predicted status
                        # Evaluate the rule using existing logic
                        alert = self._evaluate_rule(rule, customer_data)
                        if alert:
                            # Avoid duplicate open alerts for same contract and rule
                            chk = conn.cursor()
                            try:
                                # Alerts table doesn't store rule_id in this schema — check by title to avoid duplicates
                                title = f"{rule['rule_name']} - {p.get('CONTRACT_CODE') or p.get('contract_code')}"
                                chk.execute("SELECT id FROM alerts WHERE contract_code = %s AND title = %s AND status IN ('open','acknowledged') LIMIT 1", (p.get('CONTRACT_CODE') or p.get('contract_code'), title))
                                exists = chk.fetchone()
                            finally:
                                chk.close()

                            if exists:
                                skipped += 1
                                continue

                            # Ensure alert payload contains rule_id and contract_code
                            alert['rule_id'] = rule['id']
                            alert['contract_code'] = p.get('CONTRACT_CODE') or p.get('contract_code')
                            alert['branch_name'] = p.get('BRANCHNAME') or p.get('branch_name')
                            res = self.create_alert(alert)
                            if res.get('success'):
                                created += 1
                            else:
                                errors.append(res.get('error', 'unknown'))
                        else:
                            skipped += 1
                    except Exception as e:
                        errors.append(str(e))

            cursor2.close()
            return {'success': True, 'created': created, 'skipped': skipped, 'errors': errors}

        except Exception as e:
            logger.error(f"Failed to populate alerts from predictions: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()

    def populate_sme_alerts(self, since_hours: int = 24, threshold: float = 0.5) -> Dict:
        """Create SME alerts from prediction_results where SME probability >= threshold.

        - since_hours: lookback window in hours
        - threshold: SME probability threshold (0-1)
        """
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)

        created = 0
        skipped = 0
        errors = []

        try:
            # Ensure SME alert rules exist (create defaults if none)
            cursor.execute("SELECT COUNT(*) as cnt FROM alert_rules WHERE condition_type = 'sme_prediction'")
            row = cursor.fetchone()
            if not row or row.get('cnt', 0) == 0:
                self.create_default_sme_alert_rules()

            # Query recent predictions with SME probability
            sql = """
                SELECT pr.*, c.CONTRACT_CODE, c.BRANCHNAME
                FROM prediction_results pr
                JOIN customers c ON pr.customer_id = c.id
                WHERE pr.prediction_date >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            """
            cursor.execute(sql, (since_hours,))
            preds = cursor.fetchall()

            for p in preds:
                try:
                    # Extract SME probability from multiple possible column names
                    sme_prob = None
                    for key in ('sme_probability', 'SME_PROBABILITY', 'sme_prob', 'SME_PROB', 'sme_probability'):
                        if key in p and p.get(key) is not None:
                            try:
                                sme_prob = float(p.get(key))
                                break
                            except Exception:
                                continue

                    if sme_prob is None:
                        # Try reading from JSON blob 'all_probabilities' if present
                        if 'all_probabilities' in p and p.get('all_probabilities'):
                            try:
                                ap = p.get('all_probabilities')
                                if isinstance(ap, str):
                                    import json as _json
                                    ap = _json.loads(ap)
                                sme_prob = float(ap.get('SME', 0.0))
                            except Exception:
                                sme_prob = None

                    if sme_prob is None:
                        skipped += 1
                        continue

                    if float(sme_prob) < float(threshold):
                        skipped += 1
                        continue

                    contract_code = p.get('CONTRACT_CODE') or p.get('contract_code')

                    # Avoid duplicates: same contract with open/acknowledged SME alert
                    chk = conn.cursor()
                    try:
                        # Use `title` match to detect duplicates since rule_id column may not be populated
                        title = f"SME Prediction Alert - {contract_code}"
                        chk.execute("SELECT id, alert_id FROM alerts WHERE contract_code = %s AND title = %s AND status IN ('open','acknowledged') LIMIT 1", (contract_code, title))
                        exists = chk.fetchone()
                    finally:
                        chk.close()

                    # Build alert payload
                    alert_payload = {
                        'rule_id': None,
                        'contract_code': contract_code,
                        'branch_name': p.get('BRANCHNAME') or p.get('branch_name'),
                        'severity': 'medium',
                        'title': f"SME prediction alert - {contract_code}",
                        'description': f"SME probability {sme_prob:.4f} >= {threshold}",
                        'current_value': sme_prob,
                        'threshold_value': threshold,
                        'notification_channels': ['email','dashboard'],
                        'recipients': [{'role': 'branch_manager'}]
                    }

                    if exists:
                        # UPDATE existing alert with new prediction data
                        alert_id = exists.get('alert_id')
                        upd = conn.cursor()
                        try:
                            upd.execute("""
                                UPDATE alerts 
                                SET description = %s, current_value = %s, created_at = NOW()
                                WHERE alert_id = %s
                            """, (
                                alert_payload['description'],
                                sme_prob,
                                alert_id
                            ))
                            conn.commit()
                            logger.info(f"Updated SME alert {alert_id} for contract {contract_code}: SME prob={sme_prob:.4f}")
                            created += 1
                        finally:
                            upd.close()
                    else:
                        # CREATE new alert
                        res = self.create_alert(alert_payload)
                        if res.get('success'):
                            created += 1
                        else:
                            errors.append(res.get('error', 'unknown'))

                except Exception as e:
                    errors.append(str(e))

            return {'success': True, 'created': created, 'skipped': skipped, 'errors': errors}

        except Exception as e:
            logger.error(f"Failed to populate SME alerts: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def _escalate_alert(self, alert: Dict):
        """Escalate an alert to higher level"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Determine escalation target
            escalation_target = self._get_escalation_target(alert)
            
            # Update alert
            cursor.execute("""
                UPDATE alerts 
                SET status = 'escalated', escalated_at = NOW(), escalated_to = %s
                WHERE id = %s
            """, (escalation_target, alert['id']))
            
            conn.commit()
            
            # Send escalation notification
            self._send_escalation_notification(alert, escalation_target)
            
            logger.info(f"Escalated alert {alert['id']} to {escalation_target}")
            
        except Exception as e:
            logger.error(f"Failed to escalate alert: {e}")
        finally:
            cursor.close()
    
    def _get_escalation_target(self, alert: Dict) -> str:
        """Determine escalation target based on alert severity and branch"""
        severity = alert['severity']
        branch_name = alert['branch_name']
        
        if severity in ['critical', 'high']:
            # Escalate to district manager
            return f"district_manager_{self._get_district_from_branch(branch_name)}"
        else:
            # Escalate to regional manager
            return f"regional_manager_{self._get_region_from_branch(branch_name)}"
    
    def _get_district_from_branch(self, branch_name: str) -> str:
        """Get district name from branch"""
        # This would require a branch-to-district mapping
        # For now, return a default
        return "default_district"
    
    def _get_region_from_branch(self, branch_name: str) -> str:
        """Get region name from branch"""
        # This would require a branch-to-region mapping
        # For now, return a default
        return "default_region"
    
    def _send_escalation_notification(self, alert: Dict, escalation_target: str):
        """Send escalation notification"""
        # Create escalation notification
        escalation_data = {
            'alert_id': str(alert['id']),
            'title': f"ESCALATED: {alert.get('title', 'Alert')}",
            'description': f"Alert escalated due to lack of action. Original: {alert.get('description', 'No description')}",
            'severity': alert['severity'],
            'notification_channels': ['email'],
            'recipients': [{'role': escalation_target}]
        }
        
        self._send_notifications(str(alert['id']), escalation_data)
    
    def get_alerts_dashboard(self, limit: int = 100) -> Dict:
        """Get alerts data for dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Try comprehensive schema first (created_at, branch_name, etc.)
            try:
                cursor.execute("""
                    SELECT 
                        id, alert_id, contract_code, branch_name, severity, title,
                        description, current_value, threshold_value, status,
                        created_at, due_date, escalated_at
                    FROM alerts 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))

                recent_alerts = cursor.fetchall()

                # Get alert summary
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
                        SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_alerts,
                        SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_alerts,
                        SUM(CASE WHEN severity = 'low' THEN 1 ELSE 0 END) as low_alerts,
                        SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_alerts,
                        SUM(CASE WHEN status = 'escalated' THEN 1 ELSE 0 END) as escalated_alerts,
                        SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_alerts
                    FROM alerts 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)

                alert_summary = cursor.fetchone()

                # Get top branches with most alerts
                cursor.execute("""
                    SELECT 
                        branch_name,
                        COUNT(*) as alert_count,
                        SUM(CASE WHEN severity IN ('critical', 'high') THEN 1 ELSE 0 END) as high_priority_count
                    FROM alerts 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                    GROUP BY branch_name 
                    ORDER BY alert_count DESC 
                    LIMIT 10
                """)

                branch_alerts = cursor.fetchall()

                # Normalize recent_alerts to expected frontend keys
                normalized = []
                for r in recent_alerts:
                    normalized.append({
                        'alert_id': r.get('alert_id') or r.get('id') or None,
                        'contract_code': r.get('contract_code'),
                        'branch_name': r.get('branch_name') or r.get('BRANCHNAME') or None,
                        'customer_name': r.get('customer_name') or r.get('customer') or None,
                        'severity': r.get('severity'),
                        'title': r.get('title') or r.get('risk_signal') or r.get('title'),
                        'description': r.get('description') or r.get('manager_notes') or r.get('risk_signal') or '',
                        'current_value': r.get('current_value') or r.get('prediction_score') or 0.0,
                        'threshold_value': r.get('threshold_value') or None,
                        'status': r.get('status'),
                        'created_at': r.get('created_at') or r.get('alert_timestamp') or r.get('alert_timestamp')
                    })

                return {
                    'recent_alerts': normalized,
                    'alert_summary': alert_summary,
                    'branch_alerts': branch_alerts,
                    'last_updated': datetime.now().isoformat()
                }
            except Exception:
                # Fallback to minimal schema compatible queries (older alerts table)
                cursor.execute("""
                    SELECT 
                        id, contract_code, customer_name, severity, risk_signal as title,
                        prediction_score as current_value, status, alert_timestamp
                    FROM alerts
                    ORDER BY alert_timestamp DESC
                    LIMIT %s
                """, (limit,))

                recent_alerts = cursor.fetchall()

                # Minimal alert summary using available columns
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
                        SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_alerts,
                        SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_alerts,
                        SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_alerts
                    FROM alerts 
                    WHERE alert_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)

                alert_summary = cursor.fetchone()

                # Branch-level stats not available in minimal schema
                branch_alerts = []

                # Normalize minimal-schema recent_alerts
                normalized = []
                for r in recent_alerts:
                    normalized.append({
                        'alert_id': r.get('alert_id') or r.get('id') or None,
                        'contract_code': r.get('contract_code'),
                        'branch_name': None,
                        'customer_name': r.get('customer_name') or None,
                        'severity': r.get('severity'),
                        'title': r.get('title') or r.get('risk_signal') or '',
                        'description': r.get('description') or r.get('manager_notes') or r.get('risk_signal') or '',
                        'current_value': r.get('current_value') or r.get('prediction_score') or 0.0,
                        'threshold_value': None,
                        'status': r.get('status'),
                        'created_at': r.get('alert_timestamp')
                    })

                return {
                    'recent_alerts': normalized,
                    'alert_summary': alert_summary,
                    'branch_alerts': branch_alerts,
                    'last_updated': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Failed to get alerts dashboard: {e}")
            return {}
        finally:
            cursor.close()
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> Dict:
        """Acknowledge an alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE alerts 
                SET status = 'acknowledged', acknowledged_at = NOW(), acknowledged_by = %s
                WHERE alert_id = %s AND status = 'open'
            """, (user_id, alert_id))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Alert {alert_id} acknowledged by {user_id}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Alert not found or already acknowledged'}
                
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: str) -> Dict:
        """Resolve an alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE alerts 
                SET status = 'resolved', resolved_at = NOW(), resolution_notes = %s
                WHERE alert_id = %s AND status IN ('open', 'acknowledged', 'escalated')
            """, (resolution_notes, alert_id))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Alert {alert_id} resolved by {user_id}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Alert not found or already resolved'}
                
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()

    def escalate_alert(self, alert_id: str, escalated_by: Optional[str] = None) -> Dict:
        """Public wrapper to escalate an alert by id."""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute("SELECT * FROM alerts WHERE alert_id = %s LIMIT 1", (alert_id,))
            alert = cursor.fetchone()
            if not alert:
                return {'success': False, 'error': 'Alert not found'}

            # Update audit field if provided
            if escalated_by:
                try:
                    upd = conn.cursor()
                    upd.execute("UPDATE alerts SET escalated_by = %s WHERE alert_id = %s", (escalated_by, alert_id))
                    upd.close()
                except Exception:
                    pass

            # Call internal escalation handler
            self._escalate_alert(alert)

            # Ensure DAO case exists for escalated alert
            self._ensure_dao_case(alert)
            return {'success': True}
        except Exception as e:
            logger.error(f"Failed to escalate alert {alert_id}: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()

    def _ensure_dao_case(self, alert: Dict):
        """Create a DAO case from an escalated alert if no open case exists."""
        if not alert.get('contract_code'):
            return

        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT case_id FROM cases
                WHERE contract_code = %s
                AND status IN ('open', 'in_progress')
                LIMIT 1
                """,
                (alert['contract_code'],)
            )
            existing_case = cursor.fetchone()
            if existing_case:
                return

            # Create a new DAO case for the escalated alert
            case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            priority = 'high' if alert.get('severity') in ('high', 'critical') else 'medium'
            due_date = datetime.now() + timedelta(days=14 if priority == 'high' else 30)

            insert_sql = """
                INSERT INTO cases (
                    case_id, contract_code, case_type, priority, status,
                    assigned_to, assigned_by, assigned_at, due_date,
                    expected_resolution, risk_score, total_exposure, days_past_due,
                    last_payment_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_sql, (
                case_id,
                alert['contract_code'],
                'recovery',
                priority,
                'open',
                'dao',
                alert.get('escalated_by') or 'system',
                datetime.now(),
                due_date,
                None,
                alert.get('current_value', 0),
                alert.get('threshold_value', 0),
                None,
                None
            ))
            conn.commit()
            logger.info(f"Created DAO case {case_id} for escalated alert {alert.get('alert_id')}")
        except Exception as e:
            logger.error(f"Failed to create DAO case for alert {alert.get('alert_id')}: {e}")
            conn.rollback()
        finally:
            cursor.close()

# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Bant@6963',
        'database': 'lon-default'
    }
    
    # Initialize Alerts Engine
    alerts = AlertsEngine(db_config)
    
    # Create a sample SME alert rule
    rule_data = {
        'rule_name': 'SME Prediction Alert',
        'description': 'Alert when loan is predicted as SME status - requires officer attention',
        'condition_type': 'sme_prediction',
        'threshold_value': 1,  # Binary check for SME status
        'operator': '=',
        'severity': 'medium',
        'notification_channels': ['email', 'dashboard'],
        'recipients': [{'role': 'branch_manager'}],
        'escalation_hours': 48,
        'created_by': 'admin'
    }
    
    # result = alerts.create_alert_rule(rule_data)
    # print(result)
