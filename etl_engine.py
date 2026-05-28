"""
ETL Engine Module for CBE Loan Risk Management System
Handles data integration, quality checks, and anomaly detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mysql.connector
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLEngine:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        self.quality_thresholds = {
            'missing_coords_max_percentage': 5.0,
            'invalid_dates_max_percentage': 2.0,
            'amount_anomaly_threshold': 3.0  # 3 standard deviations
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
    
    def validate_date(self, date_str: str, field_name: str) -> Tuple[bool, Optional[str]]:
        """Validate date string and return (is_valid, error_message)"""
        if pd.isna(date_str) or date_str == '':
            return False, f"Missing {field_name}"
        
        try:
            parsed_date = pd.to_datetime(date_str)
            # Check for reasonable date ranges
            if parsed_date.year < 2000 or parsed_date.year > 2050:
                return False, f"{field_name} year {parsed_date.year} out of range"
            
            # Check logical date relationships
            if field_name == "GRANT_DATE" and parsed_date > datetime.now():
                return False, "GRANT_DATE cannot be in the future"
                
            return True, None
        except Exception as e:
            return False, f"Invalid {field_name} format: {e}"
    
    def validate_coordinates(self, lat: float, lon: float) -> Tuple[bool, Optional[str]]:
        """Validate latitude and longitude coordinates"""
        if pd.isna(lat) or pd.isna(lon):
            return False, "Missing coordinates"
        
        # Ethiopia approximate bounds
        if not (3.0 <= lat <= 15.0):
            return False, f"Latitude {lat} outside Ethiopia bounds"
        if not (33.0 <= lon <= 48.0):
            return False, f"Longitude {lon} outside Ethiopia bounds"
            
        return True, None
    
    def calculate_data_quality_score(self, row: pd.Series) -> float:
        """Calculate data quality score for a loan record"""
        score = 100.0
        deductions = 0
        
        # Check critical fields
        critical_fields = ['CONTRACT_CODE', 'APPROVED_AMOUNT', 'GRANT_DATE', 'EXPIRY_DATE']
        for field in critical_fields:
            if pd.isna(row.get(field)) or row.get(field) == '':
                deductions += 10
        
        # Check coordinates
        if pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
            deductions += 5
        
        # Check dates
        date_fields = ['GRANT_DATE', 'EXPIRY_DATE', 'BUSINESS_DATE']
        for field in date_fields:
            valid, _ = self.validate_date(row.get(field), field)
            if not valid:
                deductions += 5
        
        # Check financial amounts
        amount_fields = ['APPROVED_AMOUNT', 'PRINCIPAL_OS', 'INTEREST_OS']
        for field in amount_fields:
            if pd.isna(row.get(field)) or row.get(field) < 0:
                deductions += 3
        
        return max(0, score - deductions)
    
    def detect_amount_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in loan amounts using statistical methods"""
        anomalies = []
        
        # Group by branch for branch-level anomaly detection
        for branch in df['BRANCHNAME'].unique():
            if pd.isna(branch):
                continue
                
            branch_data = df[df['BRANCHNAME'] == branch]
            approved_amounts = branch_data['APPROVED_AMOUNT'].dropna()
            
            if len(approved_amounts) < 5:  # Need minimum data for statistical analysis
                continue
            
            mean_amount = approved_amounts.mean()
            std_amount = approved_amounts.std()
            
            # Detect outliers (3 standard deviations)
            for idx, row in branch_data.iterrows():
                amount = row['APPROVED_AMOUNT']
                if pd.isna(amount):
                    continue
                
                z_score = abs((amount - mean_amount) / std_amount) if std_amount > 0 else 0
                
                if z_score > self.quality_thresholds['amount_anomaly_threshold']:
                    anomalies.append({
                        'contract_code': row['CONTRACT_CODE'],
                        'branch_name': branch,
                        'metric_type': 'total_approved',
                        'current_value': amount,
                        'baseline_value': mean_amount,
                        'deviation_percentage': ((amount - mean_amount) / mean_amount) * 100,
                        'anomaly_score': z_score,
                        'is_anomaly': True,
                        'description': f'Loan amount {amount:.2f} is {z_score:.1f} std deviations from branch mean {mean_amount:.2f}'
                    })
        
        return anomalies
    
    def run_data_validation(self, df: pd.DataFrame, batch_id: str) -> Dict:
        """Run comprehensive data validation on loan data"""
        validation_results = {
            'batch_id': batch_id,
            'total_records': len(df),
            'valid_records': 0,
            'failed_records': 0,
            'validation_issues': [],
            'quality_summary': {}
        }
        
        validation_logs = []
        
        for idx, row in df.iterrows():
            contract_code = row.get('CONTRACT_CODE', f'UNKNOWN_{idx}')
            row_issues = []
            
            # Validate coordinates
            lat = row.get('Latitude')
            lon = row.get('Longitude')
            coord_valid, coord_error = self.validate_coordinates(lat, lon)
            if not coord_valid:
                row_issues.append({
                    'batch_id': batch_id,
                    'contract_code': contract_code,
                    'validation_type': 'missing_coords',
                    'severity': 'medium',
                    'description': coord_error,
                    'original_value': f'Lat: {lat}, Lon: {lon}',
                    'status': 'pending'
                })
            
            # Validate dates
            date_fields = ['GRANT_DATE', 'EXPIRY_DATE', 'BUSINESS_DATE']
            for field in date_fields:
                date_value = row.get(field)
                date_valid, date_error = self.validate_date(date_value, field)
                if not date_valid:
                    row_issues.append({
                        'batch_id': batch_id,
                        'contract_code': contract_code,
                        'validation_type': 'invalid_dates',
                        'severity': 'high',
                        'description': date_error,
                        'original_value': str(date_value),
                        'status': 'pending'
                    })
            
            # Check for missing critical fields
            critical_fields = ['CONTRACT_CODE', 'APPROVED_AMOUNT', 'LOAN_STATUS']
            for field in critical_fields:
                if pd.isna(row.get(field)) or row.get(field) == '':
                    row_issues.append({
                        'batch_id': batch_id,
                        'contract_code': contract_code,
                        'validation_type': 'missing_critical_field',
                        'severity': 'critical',
                        'description': f'Missing critical field: {field}',
                        'original_value': str(row.get(field)),
                        'status': 'pending'
                    })
            
            # Calculate data quality score
            quality_score = self.calculate_data_quality_score(row)
            
            if len(row_issues) == 0:
                validation_results['valid_records'] += 1
            else:
                validation_results['failed_records'] += 1
                validation_logs.extend(row_issues)
            
            validation_results['validation_issues'].extend(row_issues)
        
        # Store validation logs in database
        self._store_validation_logs(validation_logs)
        
        # Calculate quality summary
        validation_results['quality_summary'] = {
            'overall_quality_score': (validation_results['valid_records'] / validation_results['total_records']) * 100,
            'missing_coords_percentage': (len([log for log in validation_logs if log['validation_type'] == 'missing_coords']) / validation_results['total_records']) * 100,
            'invalid_dates_percentage': (len([log for log in validation_logs if log['validation_type'] == 'invalid_dates']) / validation_results['total_records']) * 100,
            'critical_issues_count': len([log for log in validation_logs if log['severity'] == 'critical'])
        }
        
        return validation_results
    
    def _store_validation_logs(self, validation_logs: List[Dict]):
        """Store validation logs in database"""
        if not validation_logs:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO data_validation_logs 
                (batch_id, contract_code, validation_type, severity, description, 
                 original_value, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for log in validation_logs:
                cursor.execute(sql, (
                    log['batch_id'],
                    log['contract_code'],
                    log['validation_type'],
                    log['severity'],
                    log['description'],
                    log['original_value'],
                    log['status'],
                    datetime.now()
                ))
            
            conn.commit()
            logger.info(f"Stored {len(validation_logs)} validation logs")
            
        except Exception as e:
            logger.error(f"Failed to store validation logs: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
    def store_anomalies(self, anomalies: List[Dict], detection_date: datetime):
        """Store detected anomalies in database"""
        if not anomalies:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO anomaly_detection 
                (detection_date, branch_name, metric_type, current_value, baseline_value,
                 deviation_percentage, anomaly_score, is_anomaly, description, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for anomaly in anomalies:
                cursor.execute(sql, (
                    detection_date.date(),
                    anomaly['branch_name'],
                    anomaly['metric_type'],
                    anomaly['current_value'],
                    anomaly['baseline_value'],
                    anomaly['deviation_percentage'],
                    anomaly['anomaly_score'],
                    anomaly['is_anomaly'],
                    anomaly['description'],
                    datetime.now()
                ))
            
            conn.commit()
            logger.info(f"Stored {len(anomalies)} anomalies")
            
        except Exception as e:
            logger.error(f"Failed to store anomalies: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
    def load_loan_data(self, file_path: str) -> pd.DataFrame:
        """Load loan data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def process_etl_pipeline(self, file_path: str) -> Dict:
        """Run complete ETL pipeline"""
        run_id = str(uuid.uuid4())
        batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pipeline_log = {
            'run_id': run_id,
            'batch_id': batch_id,
            'start_time': datetime.now(),
            'status': 'running',
            'records_processed': 0,
            'records_validated': 0,
            'records_failed': 0,
            'error_message': None
        }
        
        try:
            # Load data
            df = self.load_loan_data(file_path)
            pipeline_log['records_processed'] = len(df)
            
            # Run data validation
            validation_results = self.run_data_validation(df, batch_id)
            pipeline_log['records_validated'] = validation_results['valid_records']
            pipeline_log['records_failed'] = validation_results['failed_records']
            
            # Detect anomalies
            anomalies = self.detect_amount_anomalies(df)
            self.store_anomalies(anomalies, datetime.now())
            
            # Store valid records in customers table
            valid_df = df[df['CONTRACT_CODE'].isin(
                [log['contract_code'] for log in validation_results['validation_issues'] if log['severity'] == 'critical']
            ) == False]
            
            self._store_customer_data(valid_df)
            
            pipeline_log['status'] = 'completed'
            pipeline_log['end_time'] = datetime.now()
            
            # Log pipeline run
            self._log_pipeline_run(pipeline_log)
            
            return {
                'success': True,
                'run_id': run_id,
                'batch_id': batch_id,
                'validation_results': validation_results,
                'anomalies_detected': len(anomalies),
                'processing_time': (pipeline_log['end_time'] - pipeline_log['start_time']).total_seconds()
            }
            
        except Exception as e:
            pipeline_log['status'] = 'failed'
            pipeline_log['error_message'] = str(e)
            pipeline_log['end_time'] = datetime.now()
            
            self._log_pipeline_run(pipeline_log)
            
            logger.error(f"ETL Pipeline failed: {e}")
            return {
                'success': False,
                'run_id': run_id,
                'error': str(e)
            }
    
    def _store_customer_data(self, df: pd.DataFrame):
        """Store customer data in database"""
        if df.empty:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Prepare data for insertion
            records = []
            for _, row in df.iterrows():
                record = (
                    row.get('CONTRACT_CODE'),
                    row.get('CUSTOMER_ID'),
                    row.get('CO_CODE'),
                    row.get('DISTRICTNAME'),
                    row.get('REGIONNAME'),
                    row.get('CBE_REGION'),
                    row.get('BRANCHNAME'),
                    row.get('APPROVED_AMOUNT'),
                    pd.to_datetime(row.get('GRANT_DATE')) if pd.notna(row.get('GRANT_DATE')) else None,
                    pd.to_datetime(row.get('EXPIRY_DATE')) if pd.notna(row.get('EXPIRY_DATE')) else None,
                    row.get('TENURE'),
                    row.get('TERM'),
                    row.get('LOAN_TYPE'),
                    row.get('LOAN_DESCRIPTION'),
                    row.get('LOAN_PRODUCT'),
                    row.get('ARRNGEMENT_ID'),
                    row.get('RELATIONSHIP_MANAGER'),
                    row.get('LTYPE'),
                    row.get('CUST_SHORTNAME'),
                    row.get('LINE_NO'),
                    row.get('ACCT_OFFICER_CODE'),
                    row.get('DAO_NAME'),
                    row.get('DAO_CODE'),
                    row.get('PROD_CODE'),
                    pd.to_datetime(row.get('BUSINESS_DATE')) if pd.notna(row.get('BUSINESS_DATE')) else None,
                    row.get('PRINCIPAL_OS'),
                    row.get('INTEREST_OS'),
                    row.get('LOAN_STATUS'),
                    row.get('NBE_LOAN_STATUS'),
                    row.get('PRINCIPAL_ARREARS'),
                    row.get('INTEREST_ARREARS'),
                    row.get('CURRENT_COMMITTMENT'),
                    row.get('IS_GOVT_BACKED'),
                    row.get('INTEREST_RATE'),
                    row.get('INSTALLMENT_AMOUNT'),
                    row.get('INSTALLMENT_FREQ_PRINCIPAL'),
                    row.get('INSTALLMENT_FREQ_INTEREST'),
                    row.get('RISK_GRADE'),
                    pd.to_datetime(row.get('DATE_RATED')) if pd.notna(row.get('DATE_RATED')) else None,
                    row.get('ECONOMIC_SECTOR'),
                    row.get('INDUSTRY'),
                    row.get('OWNERSHIP'),
                    row.get('SECTOR'),
                    row.get('TERM_OF_PAYMENT'),
                    row.get('PRODUCT_OWNER'),
                    row.get('LOANID'),
                    row.get('COLLATTERAL'),
                    row.get('COLLATERAL_VALUE'),
                    row.get('Latitude'),
                    row.get('Longitude'),
                    row.get('AMOUNT_RANGE'),
                    row.get('COLLATERAL_RANGE'),
                    row.get('fiscal_quarter'),
                    self.calculate_data_quality_score(row),
                    pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')),
                    False,  # invalid_dates will be set during validation
                    datetime.now(),
                    datetime.now()
                )
                records.append(record)
            
            # Use INSERT ... ON DUPLICATE KEY UPDATE for upserts
            sql = """
                INSERT INTO customers (
                    CONTRACT_CODE, CUSTOMER_ID, CO_CODE, DISTRICTNAME, REGIONNAME, CBE_REGION,
                    BRANCHNAME, APPROVED_AMOUNT, GRANT_DATE, EXPIRY_DATE, TENURE, TERM,
                    LOAN_TYPE, LOAN_DESCRIPTION, LOAN_PRODUCT, ARRNGEMENT_ID, RELATIONSHIP_MANAGER,
                    LTYPE, CUST_SHORTNAME, LINE_NO, ACCT_OFFICER_CODE, DAO_NAME, DAO_CODE,
                    PROD_CODE, BUSINESS_DATE, PRINCIPAL_OS, INTEREST_OS, LOAN_STATUS,
                    NBE_LOAN_STATUS, PRINCIPAL_ARREARS, INTEREST_ARREARS, CURRENT_COMMITTMENT,
                    IS_GOVT_BACKED, INTEREST_RATE, INSTALLMENT_AMOUNT, INSTALLMENT_FREQ_PRINCIPAL,
                    INSTALLMENT_FREQ_INTEREST, RISK_GRADE, DATE_RATED, ECONOMIC_SECTOR,
                    INDUSTRY, OWNERSHIP, SECTOR, TERM_OF_PAYMENT, PRODUCT_OWNER, LOANID,
                    COLLATTERAL, COLLATERAL_VALUE, Latitude, Longitude, AMOUNT_RANGE,
                    COLLATERAL_RANGE, fiscal_quarter, DATA_QUALITY_SCORE, MISSING_COORDINATES,
                    INVALID_DATES, CREATED_AT, UPDATED_AT
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    DISTRICTNAME = VALUES(DISTRICTNAME),
                    REGIONNAME = VALUES(REGIONNAME),
                    CBE_REGION = VALUES(CBE_REGION),
                    BRANCHNAME = VALUES(BRANCHNAME),
                    APPROVED_AMOUNT = VALUES(APPROVED_AMOUNT),
                    PRINCIPAL_OS = VALUES(PRINCIPAL_OS),
                    INTEREST_OS = VALUES(INTEREST_OS),
                    PRINCIPAL_ARREARS = VALUES(PRINCIPAL_ARREARS),
                    INTEREST_ARREARS = VALUES(INTEREST_ARREARS),
                    CURRENT_COMMITTMENT = VALUES(CURRENT_COMMITTMENT),
                    LOAN_STATUS = VALUES(LOAN_STATUS),
                    DATA_QUALITY_SCORE = VALUES(DATA_QUALITY_SCORE),
                    MISSING_COORDINATES = VALUES(MISSING_COORDINATES),
                    INVALID_DATES = VALUES(INVALID_DATES),
                    UPDATED_AT = VALUES(UPDATED_AT)
            """
            
            cursor.executemany(sql, records)
            conn.commit()
            logger.info(f"Stored/updated {len(records)} customer records")
            
        except Exception as e:
            logger.error(f"Failed to store customer data: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _log_pipeline_run(self, pipeline_log: Dict):
        """Log ETL pipeline run"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = """
                INSERT INTO etl_pipeline_runs 
                (run_id, start_time, end_time, status, records_processed, 
                 records_validated, records_failed, error_message, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                pipeline_log['run_id'],
                pipeline_log['start_time'],
                pipeline_log['end_time'],
                pipeline_log['status'],
                pipeline_log['records_processed'],
                pipeline_log['records_validated'],
                pipeline_log['records_failed'],
                pipeline_log['error_message'],
                datetime.now()
            ))
            
            conn.commit()
            logger.info(f"Logged pipeline run: {pipeline_log['run_id']}")
            
        except Exception as e:
            logger.error(f"Failed to log pipeline run: {e}")
        finally:
            cursor.close()
    
    def get_data_quality_dashboard(self) -> Dict:
        """Get data quality metrics for dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get recent validation summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_validations,
                    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_issues,
                    SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_issues,
                    SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_issues,
                    SUM(CASE WHEN severity = 'low' THEN 1 ELSE 0 END) as low_issues,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_issues
                FROM data_validation_logs 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            
            validation_summary = cursor.fetchone()
            
            # Get recent anomalies
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_anomalies,
                    SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_anomalies,
                    AVG(anomaly_score) as avg_anomaly_score
                FROM anomaly_detection 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                AND is_anomaly = TRUE
            """)
            
            anomaly_summary = cursor.fetchone()
            
            # Get pipeline success rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                    AVG(records_processed) as avg_records_processed
                FROM etl_pipeline_runs 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            pipeline_summary = cursor.fetchone()
            
            return {
                'validation_summary': validation_summary,
                'anomaly_summary': anomaly_summary,
                'pipeline_summary': pipeline_summary,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get data quality dashboard: {e}")
            return {}
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
    
    # Initialize ETL Engine
    etl = ETLEngine(db_config)
    
    # Process a file (example)
    # result = etl.process_etl_pipeline("data/Loan_cbe.xlsx")
    # print(result)
