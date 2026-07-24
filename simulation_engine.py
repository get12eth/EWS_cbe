"""
What-If Simulation Module for CBE Loan Risk Management System
Handles stress testing and portfolio analysis scenarios
"""

import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np
import pandas as pd
import uuid
from scipy import stats

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationEngine:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None
        
    def get_connection(self):
        """Get database connection"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.conn = mysql.connector.connect(**self.db_config)
            return self.conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def create_simulation_scenario(self, scenario_data: Dict) -> Dict:
        """Create a new simulation scenario"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            scenario_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            sql = """
                INSERT INTO simulation_scenarios 
                (scenario_id, scenario_name, description, created_by,
                 inflation_rate, interest_rate_change, gdp_growth, unemployment_rate,
                 sector_exclusions, sector_increments, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                scenario_id,
                scenario_data['scenario_name'],
                scenario_data.get('description', ''),
                scenario_data.get('created_by', 'system'),
                scenario_data.get('inflation_rate', 0.0),
                scenario_data.get('interest_rate_change', 0.0),
                scenario_data.get('gdp_growth', 0.0),
                scenario_data.get('unemployment_rate', 0.0),
                json.dumps(scenario_data.get('sector_exclusions', [])),
                json.dumps(scenario_data.get('sector_increments', [])),
                'draft'
            ))
            
            conn.commit()
            
            logger.info(f"Created simulation scenario: {scenario_id}")
            return {'success': True, 'scenario_id': scenario_id}
            
        except Exception as e:
            logger.error(f"Failed to create simulation scenario: {e}")
            conn.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def run_simulation(self, scenario_id: str) -> Dict:
        """Run a simulation scenario"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get scenario details
            cursor.execute("SELECT * FROM simulation_scenarios WHERE scenario_id = %s", (scenario_id,))
            scenario = cursor.fetchone()
            
            if not scenario:
                return {'success': False, 'error': 'Scenario not found'}
            
            # Update status to running
            cursor.execute("UPDATE simulation_scenarios SET status = 'running' WHERE scenario_id = %s", (scenario_id,))
            conn.commit()
            
            #Get current portfolio data
            portfolio_data = self._get_current_portfolio()
            
            if not portfolio_data:
                return {'success': False, 'error': 'No portfolio data available'}
            
            # Apply economic shocks
            shocked_portfolio = self._apply_economic_shocks(portfolio_data, scenario)
            
            # Calculate sector-specific impacts
            sector_results = self._calculate_sector_impacts(shocked_portfolio, scenario)
            
            # Calculate overall portfolio impact
            portfolio_impact = self._calculate_portfolio_impact(sector_results)
            
            # Store results
            self._store_simulation_results(scenario_id, sector_results, portfolio_impact)
            
            # Update scenario with results
            cursor.execute("""
                UPDATE simulation_scenarios 
                SET status = 'completed', completed_at = NOW(),
                    total_portfolio_value = %s, predicted_npl_count = %s,
                    predicted_npl_percentage = %s, risk_adjusted_return = %s
                WHERE scenario_id = %s
            """, (
                portfolio_impact['total_portfolio_value'],
                portfolio_impact['predicted_npl_count'],
                portfolio_impact['predicted_npl_percentage'],
                portfolio_impact['risk_adjusted_return'],
                scenario_id
            ))
            
            conn.commit()
            
            logger.info(f"Simulation completed: {scenario_id}")
            return {
                'success': True,
                'scenario_id': scenario_id,
                'results': {
                    'sector_results': sector_results,
                    'portfolio_impact': portfolio_impact
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to run simulation: {e}")
            cursor.execute("UPDATE simulation_scenarios SET status = 'failed' WHERE scenario_id = %s", (scenario_id,))
            conn.commit()
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def _get_current_portfolio(self) -> Optional[pd.DataFrame]:
        """Get current portfolio data"""
        try:
            conn = self.get_connection()
            
            query = """
                SELECT CONTRACT_CODE, APPROVED_AMOUNT, PRINCIPAL_OS, ECONOMIC_SECTOR,
                       NPL_PROBABILITY, PREDICTED_STATUS, LOAN_STATUS, BRANCHNAME,
                       BUSINESS_DATE, GRANT_DATE, EXPIRY_DATE, LOAN_AGE_DAYS
                FROM customers 
                WHERE PREDICTED_STATUS IS NOT NULL
                AND PRINCIPAL_OS > 0
            """
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return None
            
            # Calculate additional fields
            df['remaining_days'] = (pd.to_datetime(df['EXPIRY_DATE']) - pd.to_datetime(df['BUSINESS_DATE'])).dt.days
            df['loan_to_value_ratio'] = df['PRINCIPAL_OS'] / df['APPROVED_AMOUNT']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get current portfolio: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _apply_economic_shocks(self, portfolio: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """Apply economic shocks to portfolio"""
        shocked_portfolio = portfolio.copy()
        
        #Extract scenario parameters
        inflation_rate = scenario.get('inflation_rate', 0.0) / 100
        interest_rate_change = scenario.get('interest_rate_change', 0.0) / 100
        gdp_growth = scenario.get('gdp_growth', 0.0) / 100
        unemployment_rate = scenario.get('unemployment_rate', 0.0) / 100
        
        #Apply macroeconomic adjustments to NPL probabilities
        base_npl_prob = shocked_portfolio['NPL_PROBABILITY'].copy()
        
        #Inflation impact (higher inflation may increase NPL risk)
        inflation_impact = 1 + (inflation_rate * 0.5)  # Moderate sensitivity
        
        #Interest rate impact (higher rates increase default risk)
        interest_impact = 1 + (interest_rate_change * 1.2)  # High sensitivity
        
        #GDP growth impact (negative GDP increases risk)
        gdp_impact = 1 - (gdp_growth * 0.8)  # Negative correlation
        
        # Unemployment impact (higher unemployment increases risk)
        unemployment_impact = 1 + (unemployment_rate * 1.5)  # High sensitivity
        
        #Combined impact
        combined_impact = inflation_impact * interest_impact * gdp_impact * unemployment_impact
        
        # Apply to NPL probabilities with bounds
        shocked_npl_prob = base_npl_prob * combined_impact
        shocked_npl_prob = np.clip(shocked_npl_prob, 0.0, 1.0)
        
        shocked_portfolio['shocked_npl_probability'] = shocked_npl_prob
        
        # Recalculate predicted status based on shocked probabilities
        shocked_portfolio['shocked_predicted_status'] = np.where(
            shocked_npl_prob > 0.5, 'NPL', 'PAS'
        )
        
        return shocked_portfolio
    
    def _calculate_sector_impacts(self, portfolio: pd.DataFrame, scenario: Dict) -> List[Dict]:
        """Calculate sector-specific impacts"""
        sector_results = []
        
        #Get scenario parameters
        sector_exclusions = json.loads(scenario.get('sector_exclusions', '[]'))
        sector_increments = json.loads(scenario.get('sector_increments', '[]'))
        
        #Group by sector
        sector_groups = portfolio.groupby('ECONOMIC_SECTOR')
        
        for sector_name, sector_data in sector_groups:
            # Skip excluded sectors
            if sector_name in sector_exclusions:
                continue
            
            #Apply sector increments
            sector_multiplier = 1.0
            for increment in sector_increments:
                if increment.get('sector') == sector_name:
                    sector_multiplier = 1 + (increment.get('increase_percentage', 0) / 100)
                    break
            
            #Calculate baseline metrics
            baseline_loan_count = len(sector_data)
            baseline_portfolio_value = sector_data['PRINCIPAL_OS'].sum()
            baseline_npl_count = (sector_data['PREDICTED_STATUS'] == 'NPL').sum()
            baseline_npl_rate = baseline_npl_count / baseline_loan_count if baseline_loan_count > 0 else 0
            
            #Calculate simulated metrics
            simulated_loan_count = int(baseline_loan_count * sector_multiplier)
            simulated_portfolio_value = baseline_portfolio_value * sector_multiplier
            
            #Use shocked probabilities for simulation
            simulated_npl_count = (sector_data['shocked_predicted_status'] == 'NPL').sum()
            simulated_npl_rate = simulated_npl_count / simulated_loan_count if simulated_loan_count > 0 else 0
            
            #Calculate impacts
            npl_change = simulated_npl_count - baseline_npl_count
            npl_rate_change = simulated_npl_rate - baseline_npl_rate
            portfolio_value_change = simulated_portfolio_value - baseline_portfolio_value
            
            sector_result = {
                'economic_sector': sector_name,
                'baseline_loan_count': baseline_loan_count,
                'baseline_portfolio_value': baseline_portfolio_value,
                'baseline_npl_count': baseline_npl_count,
                'baseline_npl_rate': baseline_npl_rate,
                'simulated_loan_count': simulated_loan_count,
                'simulated_portfolio_value': simulated_portfolio_value,
                'simulated_npl_count': simulated_npl_count,
                'simulated_npl_rate': simulated_npl_rate,
                'npl_change': npl_change,
                'npl_rate_change': npl_rate_change,
                'portfolio_value_change': portfolio_value_change,
                'sector_multiplier': sector_multiplier
            }
            
            sector_results.append(sector_result)
        
        return sector_results
    
    def _calculate_portfolio_impact(self, sector_results: List[Dict]) -> Dict:
        """Calculate overall portfolio impact"""
        total_baseline_value = sum(r['baseline_portfolio_value'] for r in sector_results)
        total_simulated_value = sum(r['simulated_portfolio_value'] for r in sector_results)
        total_baseline_npl = sum(r['baseline_npl_count'] for r in sector_results)
        total_simulated_npl = sum(r['simulated_npl_count'] for r in sector_results)
        total_baseline_loans = sum(r['baseline_loan_count'] for r in sector_results)
        total_simulated_loans = sum(r['simulated_loan_count'] for r in sector_results)
        
        portfolio_impact = {
            'total_portfolio_value': total_simulated_value,
            'predicted_npl_count': total_simulated_npl,
            'predicted_npl_percentage': total_simulated_npl / total_simulated_loans if total_simulated_loans > 0 else 0,
            'baseline_npl_percentage': total_baseline_npl / total_baseline_loans if total_baseline_loans > 0 else 0,
            'npl_count_change': total_simulated_npl - total_baseline_npl,
            'npl_percentage_change': (total_simulated_npl / total_simulated_loans) - (total_baseline_npl / total_baseline_loans) if total_simulated_loans > 0 and total_baseline_loans > 0 else 0,
            'portfolio_value_change': total_simulated_value - total_baseline_value,
            'portfolio_value_change_percentage': ((total_simulated_value - total_baseline_value) / total_baseline_value * 100) if total_baseline_value > 0 else 0,
            'risk_adjusted_return': self._calculate_risk_adjusted_return(sector_results)
        }
        
        return portfolio_impact
    
    def _calculate_risk_adjusted_return(self, sector_results: List[Dict]) -> float:
        """Calculate risk-adjusted return for the portfolio"""
        # Simplified risk-adjusted return calculation
        # In practice, this would consider interest income, expected losses, risk premiums, etc.
        
        total_return = 0.0
        total_risk = 0.0
        
        for result in sector_results:
            # Simplified return calculation (would be more complex in reality)
            sector_return = result['simulated_portfolio_value'] * 0.12  # Assume 12% gross return
            sector_risk = result['simulated_npl_count'] * 50000  # Assume $50k loss per NPL
            
            total_return += sector_return
            total_risk += sector_risk
        
        if total_risk > 0:
            return (total_return - total_risk) / total_risk
        else:
            return 0.0
    
    def _store_simulation_results(self, scenario_id: str, sector_results: List[Dict], portfolio_impact: Dict):
        """Store simulation results in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            #Store sector results
            sql = """
                INSERT INTO simulation_sector_results 
                (scenario_id, economic_sector, baseline_loan_count, baseline_portfolio_value,
                 baseline_npl_count, baseline_npl_rate, simulated_loan_count, 
                 simulated_portfolio_value, simulated_npl_count, simulated_npl_rate,
                 npl_change, npl_rate_change, portfolio_value_change)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for result in sector_results:
                cursor.execute(sql, (
                    scenario_id,
                    result['economic_sector'],
                    result['baseline_loan_count'],
                    result['baseline_portfolio_value'],
                    result['baseline_npl_count'],
                    result['baseline_npl_rate'],
                    result['simulated_loan_count'],
                    result['simulated_portfolio_value'],
                    result['simulated_npl_count'],
                    result['simulated_npl_rate'],
                    result['npl_change'],
                    result['npl_rate_change'],
                    result['portfolio_value_change']
                ))
            
            conn.commit()
            logger.info(f"Stored simulation results for {scenario_id}")
            
        except Exception as e:
            logger.error(f"Failed to store simulation results: {e}")
            conn.rollback()
        finally:
            cursor.close()
    
    def get_simulation_results(self, scenario_id: str) -> Optional[Dict]:
        """Get simulation results"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get scenario details
            cursor.execute("SELECT * FROM simulation_scenarios WHERE scenario_id = %s", (scenario_id,))
            scenario = cursor.fetchone()
            
            if not scenario:
                return None
            
            #Get sector results
            cursor.execute("""
                SELECT * FROM simulation_sector_results 
                WHERE scenario_id = %s 
                ORDER BY simulated_npl_rate DESC
            """, (scenario_id,))
            
            sector_results = cursor.fetchall()
            
            return {
                'scenario': scenario,
                'sector_results': sector_results
            }
            
        except Exception as e:
            logger.error(f"Failed to get simulation results: {e}")
            return None
        finally:
            cursor.close()
    
    def compare_scenarios(self, scenario_ids: List[str]) -> Dict:
        """Compare multiple simulation scenarios"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            scenarios_data = []
            
            for scenario_id in scenario_ids:
                # Get scenario summary
                cursor.execute("""
                    SELECT scenario_id, scenario_name, status, total_portfolio_value,
                           predicted_npl_count, predicted_npl_percentage, risk_adjusted_return
                    FROM simulation_scenarios 
                    WHERE scenario_id = %s AND status = 'completed'
                """, (scenario_id,))
                
                scenario = cursor.fetchone()
                if scenario:
                    scenarios_data.append(scenario)
            
            # Get current portfolio for baseline comparison
            cursor.execute("""
                SELECT 
                    SUM(PRINCIPAL_OS) as total_portfolio_value,
                    COUNT(*) as total_loans,
                    SUM(CASE WHEN PREDICTED_STATUS = 'NPL' THEN 1 ELSE 0 END) as npl_count
                FROM customers 
                WHERE PREDICTED_STATUS IS NOT NULL
            """)
            
            baseline = cursor.fetchone()
            baseline_npl_percentage = baseline['npl_count'] / baseline['total_loans'] if baseline['total_loans'] > 0 else 0
            
            # Calculate comparison metrics
            comparison = {
                'baseline': {
                    'total_portfolio_value': baseline['total_portfolio_value'],
                    'predicted_npl_count': baseline['npl_count'],
                    'predicted_npl_percentage': baseline_npl_percentage
                },
                'scenarios': scenarios_data
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare scenarios: {e}")
            return {}
        finally:
            cursor.close()
    
    def get_simulation_dashboard(self) -> Dict:
        """Get simulation data for dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Get recent simulations
            cursor.execute("""
                SELECT scenario_id, scenario_name, status, created_at, completed_at,
                       total_portfolio_value, predicted_npl_count, predicted_npl_percentage
                FROM simulation_scenarios 
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            recent_simulations = cursor.fetchall()
            
            # Get simulation summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_simulations,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_simulations,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_simulations,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_simulations
                FROM simulation_scenarios 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            simulation_summary = cursor.fetchone()
            
            # Get top sector impacts from latest completed simulation
            cursor.execute("""
                SELECT ss.economic_sector, ss.simulated_npl_rate, ss.npl_rate_change,
                       ss.portfolio_value_change, s.scenario_name
                FROM simulation_sector_results ss
                JOIN simulation_scenarios s ON ss.scenario_id = s.scenario_id
                WHERE s.status = 'completed'
                AND s.scenario_id = (
                    SELECT scenario_id FROM simulation_scenarios 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC 
                    LIMIT 1
                )
                ORDER BY ABS(ss.npl_rate_change) DESC
                LIMIT 10
            """)
            
            sector_impacts = cursor.fetchall()
            
            return {
                'recent_simulations': recent_simulations,
                'simulation_summary': simulation_summary,
                'sector_impacts': sector_impacts,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get simulation dashboard: {e}")
            return {}
        finally:
            cursor.close()
    
    def create_stress_test_scenarios(self) -> List[Dict]:
        """Create predefined stress test scenarios"""
        scenarios = [
            {
                'scenario_name': 'Economic Recession',
                'description': 'Simulate economic recession conditions',
                'inflation_rate': 8.0,
                'interest_rate_change': 2.0,
                'gdp_growth': -3.0,
                'unemployment_rate': 12.0,
                'sector_exclusions': [],
                'sector_increments': []
            },
            {
                'scenario_name': 'High Interest Rate Environment',
                'description': 'Simulate high interest rate environment',
                'inflation_rate': 10.0,
                'interest_rate_change': 5.0,
                'gdp_growth': 1.0,
                'unemployment_rate': 8.0,
                'sector_exclusions': [],
                'sector_increments': []
            },
            {
                'scenario_name': 'Sector Shock - Construction',
                'description': 'Simulate construction sector crisis',
                'inflation_rate': 5.0,
                'interest_rate_change': 1.0,
                'gdp_growth': 0.0,
                'unemployment_rate': 7.0,
                'sector_exclusions': ['Construction'],
                'sector_increments': [
                    {'sector': 'Agriculture', 'increase_percentage': 20},
                    {'sector': 'Manufacturing', 'increase_percentage': 15}
                ]
            },
            {
                'scenario_name': 'Optimistic Growth',
                'description': 'Simulate optimistic economic growth',
                'inflation_rate': 3.0,
                'interest_rate_change': -1.0,
                'gdp_growth': 5.0,
                'unemployment_rate': 4.0,
                'sector_exclusions': [],
                'sector_increments': []
            }
        ]
        
        created_scenarios = []
        for scenario_data in scenarios:
            result = self.create_simulation_scenario(scenario_data)
            if result['success']:
                created_scenarios.append(result['scenario_id'])
        
        return created_scenarios

#Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Bant@6963',
        'database': 'lon-default'
    }
    
    #Initialize Simulation Engine
    sim_engine = SimulationEngine(db_config)
    
    #Create a stress test scenario
    scenario_data = {
        'scenario_name': 'COVID-19 Impact',
        'description': 'Simulate pandemic economic impact',
        'inflation_rate': 6.0,
        'interest_rate_change': 1.5,
        'gdp_growth': -2.0,
        'unemployment_rate': 10.0,
        'sector_exclusions': ['Tourism', 'Hospitality'],
        'sector_increments': [
            {'sector': 'Technology', 'increase_percentage': 25},
            {'sector': 'Healthcare', 'increase_percentage': 20}
        ],
        'created_by': 'admin'
    }
    
    #result = sim_engine.create_simulation_scenario(scenario_data)
    #if result['success']:
    #sim_result = sim_engine.run_simulation(result['scenario_id'])
    #print(sim_result)
