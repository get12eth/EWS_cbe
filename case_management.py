"""
Case Management & Workflow Module for CBE Loan Risk Management System
Handles loan recovery cases, task assignment, and resolution tracking
"""

import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaseManagement:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        self.case_types = {
            'recovery': 'Loan Recovery Case',
            'restructuring': 'Loan Restructuring Case',
            'write_off': 'Write-off Case',
            'investigation': 'Fraud Investigation Case'
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
    
    def create_case(self, case_data: Dict) -> Dict:
        """Create a new case for a high-risk loan"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Determine priority based on risk score and amount
            priority = self._determine_case_priority(case_data)
            
            # Calculate due date based on priority
            due_days = self._get_due_days_by_priority(priority)
            due_date = datetime.now() + timedelta(days=due_days)
            
            sql = """
                INSERT INTO cases 
                (case_id, contract_code, case_type, priority, status, assigned_to, 
                 assigned_by, assigned_at, due_date, expected_resolution, 
                 risk_score, total_exposure, days_past_due, last_payment_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                case_id,
                case_data['contract_code'],
                case_data['case_type'],
                priority,
                'open',
                case_data.get('assigned_to'),
                case_data.get('assigned_by', 'system'),
                datetime.now(),
                due_date,
                case_data.get('expected_resolution'),
                case_data.get('risk_score', 0),
                case_data.get('total_exposure', 0),
                case_data.get('days_past_due', 0),
                case_data.get('last_payment_date')
            ))
            
            conn.commit()
            
            logger.info(f"Created case: {case_id} for contract {case_data['contract_code']}")
            return {'success': True, 'case_id': case_id}
            
        except Exception as e:
            logger.error(f"Failed to create case: {e}")
            conn.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def _determine_case_priority(self, case_data: Dict) -> str:
        """Determine case priority based on risk factors"""
        risk_score = case_data.get('risk_score', 0)
        total_exposure = case_data.get('total_exposure', 0)
        days_past_due = case_data.get('days_past_due', 0)
        
        # Priority scoring matrix
        if risk_score >= 0.8 or total_exposure >= 1000000 or days_past_due >= 180:
            return 'urgent'
        elif risk_score >= 0.6 or total_exposure >= 500000 or days_past_due >= 90:
            return 'high'
        elif risk_score >= 0.4 or total_exposure >= 100000 or days_past_due >= 30:
            return 'medium'
        else:
            return 'low'
    
    def _get_due_days_by_priority(self, priority: str) -> int:
        """Get due days based on priority"""
        due_days_map = {
            'urgent': 7,
            'high': 14,
            'medium': 30,
            'low': 60
        }
        return due_days_map.get(priority, 30)
    
    def assign_case(self, case_id: str, assigned_to: str, assigned_by: str) -> Dict:
        """Assign a case to a recovery officer"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE cases 
                SET assigned_to = %s, assigned_by = %s, assigned_at = NOW(), status = 'in_progress'
                WHERE case_id = %s AND status = 'open'
            """, (assigned_to, assigned_by, case_id))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Case {case_id} assigned to {assigned_to}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Case not found or already assigned'}
                
        except Exception as e:
            logger.error(f"Failed to assign case: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def add_case_activity(self, case_id: str, activity_data: Dict, user_id: str) -> Dict:
        """Add activity log to a case"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO case_activities 
                (case_id, activity_type, description, outcome, next_action, 
                 next_action_date, amount, created_by, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                case_id,
                activity_data['activity_type'],
                activity_data.get('description', ''),
                activity_data.get('outcome', ''),
                activity_data.get('next_action', ''),
                activity_data.get('next_action_date'),
                activity_data.get('amount'),
                user_id,
                datetime.now()
            ))
            
            conn.commit()
            
            logger.info(f"Added activity to case {case_id}: {activity_data['activity_type']}")

            # Log audit event for activity
            try:
                self.log_audit_event('case_activity', 'case', case_id, user_id, activity_data)
            except Exception:
                pass
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Failed to add case activity: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def resolve_case(self, case_id: str, resolution_data: Dict, user_id: str) -> Dict:
        """Resolve a case"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE cases 
                SET status = 'resolved', resolution_type = %s, resolution_amount = %s,
                    resolution_date = %s, resolution_notes = %s, updated_at = NOW()
                WHERE case_id = %s AND status IN ('open', 'in_progress')
            """, (
                resolution_data['resolution_type'],
                resolution_data.get('resolution_amount', 0),
                resolution_data.get('resolution_date', datetime.now().date()),
                resolution_data.get('resolution_notes', ''),
                case_id
            ))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Case {case_id} resolved by {user_id}")
                try:
                    self.log_audit_event('case_resolved', 'case', case_id, user_id, resolution_data)
                except Exception:
                    pass
                return {'success': True}
            else:
                return {'success': False, 'error': 'Case not found or already resolved'}
                
        except Exception as e:
            logger.error(f"Failed to resolve case: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def get_case_details(self, case_id: str) -> Optional[Dict]:
        """Get detailed case information"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get case details
            cursor.execute("SELECT * FROM cases WHERE case_id = %s", (case_id,))
            case_details = cursor.fetchone()
            
            if not case_details:
                return None
            
            # Get case activities
            cursor.execute("""
                SELECT * FROM case_activities 
                WHERE case_id = %s 
                ORDER BY created_at DESC
            """, (case_id,))
            
            activities = cursor.fetchall()
            
            # Get customer information
            cursor.execute("""
                SELECT CONTRACT_CODE, CUST_SHORTNAME, APPROVED_AMOUNT, PRINCIPAL_OS,
                       BRANCHNAME, LOAN_STATUS, ECONOMIC_SECTOR
                FROM customers 
                WHERE CONTRACT_CODE = %s
            """, (case_details['contract_code'],))
            
            customer_info = cursor.fetchone()
            
            return {
                'case_details': case_details,
                'activities': activities,
                'customer_info': customer_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get case details: {e}")
            return None
        finally:
            cursor.close()
    
    def get_my_cases(self, user_id: str, status: Optional[str] = None) -> List[Dict]:
        """Get cases assigned to a user"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            sql = """
                SELECT c.*, cu.CUST_SHORTNAME, cu.APPROVED_AMOUNT, cu.BRANCHNAME
                FROM cases c
                LEFT JOIN customers cu ON c.contract_code = cu.CONTRACT_CODE
                WHERE c.assigned_to = %s
            """
            params = [user_id]
            
            if status:
                sql += " AND c.status = %s"
                params.append(status)
            
            sql += " ORDER BY c.created_at DESC"
            
            cursor.execute(sql, params)
            cases = cursor.fetchall()
            
            # Add activity count for each case
            for case in cases:
                cursor.execute("""
                    SELECT COUNT(*) as activity_count 
                    FROM case_activities 
                    WHERE case_id = %s
                """, (case['case_id'],))
                
                activity_count = cursor.fetchone()
                case['activity_count'] = activity_count['activity_count']
            
            return cases
            
        except Exception as e:
            logger.error(f"Failed to get user cases: {e}")
            return []
        finally:
            cursor.close()
    
    def get_cases_dashboard(self, user_role: str = None, user_branch: str = None) -> Dict:
        """Get cases data for dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Build WHERE clause based on user role and branch
            where_clause = ""
            params = []
            
            if user_role == 'branch_manager' and user_branch:
                where_clause = "WHERE cu.BRANCHNAME = %s"
                params.append(user_branch)
            elif user_role == 'recovery_officer':
                where_clause = "WHERE c.assigned_to = %s"
                params.append(user_branch)  # user_branch contains user_id in this context
            
            # Get case summary
            sql = f"""
                SELECT 
                    COUNT(*) as total_cases,
                    SUM(CASE WHEN priority = 'urgent' THEN 1 ELSE 0 END) as urgent_cases,
                    SUM(CASE WHEN priority = 'high' THEN 1 ELSE 0 END) as high_cases,
                    SUM(CASE WHEN priority = 'medium' THEN 1 ELSE 0 END) as medium_cases,
                    SUM(CASE WHEN priority = 'low' THEN 1 ELSE 0 END) as low_cases,
                    SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_cases,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress_cases,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_cases,
                    SUM(CASE WHEN due_date < NOW() AND status IN ('open', 'in_progress') THEN 1 ELSE 0 END) as overdue_cases,
                    SUM(CASE WHEN status = 'resolved' THEN resolution_amount ELSE 0 END) as total_recovered
                FROM cases c
                LEFT JOIN customers cu ON c.contract_code = cu.CONTRACT_CODE
                {where_clause}
            """
            
            cursor.execute(sql, params)
            case_summary = cursor.fetchone()
            
            # Get recent cases
            sql = f"""
                SELECT c.case_id, c.contract_code, c.priority, c.status, c.assigned_to,
                       c.created_at, c.due_date, cu.CUST_SHORTNAME, cu.APPROVED_AMOUNT,
                       cu.BRANCHNAME, cu.LOAN_STATUS
                FROM cases c
                LEFT JOIN customers cu ON c.contract_code = cu.CONTRACT_CODE
                {where_clause}
                ORDER BY c.created_at DESC
                LIMIT 20
            """
            
            cursor.execute(sql, params)
            recent_cases = cursor.fetchall()
            
            # Get top recovery officers
            cursor.execute("""
                SELECT 
                    assigned_to,
                    COUNT(*) as total_cases,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_cases,
                    SUM(CASE WHEN status = 'resolved' THEN resolution_amount ELSE 0 END) as total_recovered
                FROM cases 
                WHERE assigned_to IS NOT NULL
                GROUP BY assigned_to
                ORDER BY resolved_cases DESC
                LIMIT 10
            """)
            
            top_officers = cursor.fetchall()
            
            # Get cases by case type
            sql = f"""
                SELECT 
                    case_type,
                    COUNT(*) as case_count,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_count,
                    AVG(total_exposure) as avg_exposure
                FROM cases c
                LEFT JOIN customers cu ON c.contract_code = cu.CONTRACT_CODE
                {where_clause}
                GROUP BY case_type
                ORDER BY case_count DESC
            """
            
            cursor.execute(sql, params)
            cases_by_type = cursor.fetchall()
            
            return {
                'case_summary': case_summary,
                'recent_cases': recent_cases,
                'top_officers': top_officers,
                'cases_by_type': cases_by_type,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cases dashboard: {e}")
            return {}
        finally:
            cursor.close()
    
    def get_overdue_cases(self) -> List[Dict]:
        """Get overdue cases for escalation"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT c.*, cu.CUST_SHORTNAME, cu.APPROVED_AMOUNT, cu.BRANCHNAME,
                       DATEDIFF(NOW(), c.due_date) as days_overdue
                FROM cases c
                LEFT JOIN customers cu ON c.contract_code = cu.CONTRACT_CODE
                WHERE c.due_date < NOW() 
                AND c.status IN ('open', 'in_progress')
                ORDER BY days_overdue DESC
            """)
            
            overdue_cases = cursor.fetchall()
            return overdue_cases
            
        except Exception as e:
            logger.error(f"Failed to get overdue cases: {e}")
            return []
        finally:
            cursor.close()
    
    def auto_create_cases_from_alerts(self):
        """Automatically create cases from high-priority alerts"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get high-priority alerts that don't have cases yet
            cursor.execute("""
                SELECT a.*, cu.NPL_PROBABILITY, cu.PRINCIPAL_OS, cu.BRANCHNAME,
                       DATEDIFF(NOW(), cu.BUSINESS_DATE) as days_past_due
                FROM alerts a
                LEFT JOIN customers cu ON a.contract_code = cu.CONTRACT_CODE
                WHERE a.severity IN ('critical', 'high')
                AND a.status = 'open'
                AND a.contract_code NOT IN (
                    SELECT contract_code FROM cases WHERE status IN ('open', 'in_progress')
                )
            """)
            
            alerts_for_cases = cursor.fetchall()
            created_cases = 0
            
            for alert in alerts_for_cases:
                case_data = {
                    'contract_code': alert['contract_code'],
                    'case_type': 'recovery',
                    'assigned_to': self._get_best_recovery_officer(alert['branch_name']),
                    'assigned_by': 'system',
                    'risk_score': alert.get('current_value', 0),
                    'total_exposure': alert.get('PRINCIPAL_OS', 0),
                    'days_past_due': alert.get('days_past_due', 0),
                    'last_payment_date': None  # Would need to be calculated
                }
                
                result = self.create_case(case_data)
                if result['success']:
                    created_cases += 1
                    # Update alert to link to case
                    self._link_alert_to_case(alert['alert_id'], result['case_id'])
            
            logger.info(f"Auto-created {created_cases} cases from alerts")
            return created_cases
            
        except Exception as e:
            logger.error(f"Failed to auto-create cases: {e}")
            return 0
        finally:
            cursor.close()
    
    def _get_best_recovery_officer(self, branch_name: str) -> str:
        """Get the best recovery officer for a branch"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get recovery officer with least cases for the branch
            cursor.execute("""
                SELECT assigned_to, COUNT(*) as case_count
                FROM cases 
                WHERE assigned_to IN (
                    SELECT username FROM users 
                    WHERE role = 'recovery_officer' 
                    AND (branch = %s OR branch IS NULL)
                    AND is_active = TRUE
                )
                AND status IN ('open', 'in_progress')
                GROUP BY assigned_to
                ORDER BY case_count ASC
                LIMIT 1
            """, (branch_name,))
            
            result = cursor.fetchone()
            return result[0] if result else 'unassigned'
            
        except Exception:
            return 'unassigned'
        finally:
            cursor.close()
    
    def _link_alert_to_case(self, alert_id: str, case_id: str):
        """Link an alert to a case"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # This would require adding a case_id column to alerts table
            # For now, just log the linkage
            logger.info(f"Linked alert {alert_id} to case {case_id}")
            
        except Exception as e:
            logger.error(f"Failed to link alert to case: {e}")
        finally:
            cursor.close()
    
    def get_case_performance_metrics(self, days: int = 30) -> Dict:
        """Get case performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Resolution rates by priority
            cursor.execute("""
                SELECT 
                    priority,
                    COUNT(*) as total_cases,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_cases,
                    AVG(DATEDIFF(resolved_at, assigned_at)) as avg_resolution_days,
                    SUM(CASE WHEN status = 'resolved' THEN resolution_amount ELSE 0 END) as total_recovered
                FROM cases 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY priority
                ORDER BY priority
            """, (days,))
            
            resolution_by_priority = cursor.fetchall()
            
            # Officer performance
            cursor.execute("""
                SELECT 
                    assigned_to,
                    COUNT(*) as total_cases,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_cases,
                    AVG(DATEDIFF(resolved_at, assigned_at)) as avg_resolution_days,
                    SUM(CASE WHEN status = 'resolved' THEN resolution_amount ELSE 0 END) as total_recovered
                FROM cases 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                AND assigned_to IS NOT NULL
                GROUP BY assigned_to
                HAVING total_cases >= 3
                ORDER BY resolved_cases DESC
            """, (days,))
            
            officer_performance = cursor.fetchall()
            
            # Recovery trends
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as cases_created,
                    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as cases_resolved,
                    SUM(CASE WHEN status = 'resolved' THEN resolution_amount ELSE 0 END) as amount_recovered
                FROM cases 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY DATE(created_at)
                ORDER BY date
            """, (days,))
            
            recovery_trends = cursor.fetchall()
            
            return {
                'resolution_by_priority': resolution_by_priority,
                'officer_performance': officer_performance,
                'recovery_trends': recovery_trends,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Failed to get case performance metrics: {e}")
            return {}
        finally:
            cursor.close()

    def log_remediation_action(self, case_id: str, action_data: Dict, user_id: str) -> Dict:
        """Log a remediation action and create a corresponding case activity and audit event"""
        try:
            # Add as a case activity of type 'remediation'
            activity = {
                'activity_type': action_data.get('action_type', 'remediation'),
                'description': action_data.get('description', ''),
                'outcome': action_data.get('outcome', ''),
                'next_action': action_data.get('next_action', ''),
                'next_action_date': action_data.get('next_action_date'),
                'amount': action_data.get('amount')
            }

            res = self.add_case_activity(case_id, activity, user_id)

            # Log audit event
            try:
                self.log_audit_event('remediation_action', 'case', case_id, user_id, action_data)
            except Exception:
                pass

            return res
        except Exception as e:
            logger.error(f"Failed to log remediation action: {e}")
            return {'success': False, 'error': str(e)}

    def log_audit_event(self, event_type: str, object_type: str, object_id: str, performed_by: str, details: Dict) -> None:
        """Write an audit log entry to `audit_logs` table"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO audit_logs (event_type, object_type, object_id, performed_by, details)
                VALUES (%s, %s, %s, %s, %s)
            """, (event_type, object_type, object_id, performed_by, json.dumps(details)))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
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
    
    # Initialize Case Management
    case_mgmt = CaseManagement(db_config)
    
    # Create a sample case
    case_data = {
        'contract_code': 'C000001',
        'case_type': 'recovery',
        'assigned_to': 'recovery_officer_1',
        'assigned_by': 'system',
        'risk_score': 0.85,
        'total_exposure': 500000,
        'days_past_due': 120
    }
    
    # result = case_mgmt.create_case(case_data)
    # print(result)
