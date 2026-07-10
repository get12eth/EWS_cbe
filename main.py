from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import mysql.connector
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware
import logging
from typing import List, Dict, Optional
import json
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import threading
import time

#Import new modules
from etl_engine import ETLEngine
from alerts_engine import AlertsEngine
from case_management import CaseManagement
from model_governance import ModelGovernance
from simulation_engine import SimulationEngine

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Optional SHAP support
try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    shap = None
    _SHAP_AVAILABLE = False

_SHAP_EXPLAINER = None

app = FastAPI(title="CBE Loan Risk Dashboard", debug=True)

#Mount static files only if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.cache_size = 0

def get_user_role(request: Request) -> Optional[str]:
    username = request.session.get('user')
    if not username:
        return None
    info = _get_user_info(username)
    return info.get('role') if info else None

templates.env.globals['get_user_role'] = get_user_role

#Password hashing context
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

#Database connection helper (uses local MySQL with provided root password)
def get_db_connection():
    return mysql.connector.connect(host='localhost', user='root', password='Bant@6963', database='lon-default')

def _get_user_info(username: str) -> Optional[Dict]:
    """Return user record from users table or None if unavailable."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT username, role, is_active FROM users WHERE username = %s LIMIT 1", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Exception:
        return None

def _require_roles(request: Request, allowed_roles: List[str]) -> Optional[Dict]:
    """Check session user role against allowed_roles. Returns user dict if allowed, otherwise None."""
    user = request.session.get('user')
    if not user:
        return None
    info = _get_user_info(user)
    if not info or not info.get('is_active'):
        return None
    if info.get('role') in allowed_roles:
        return info
    return None


def _redirect_dao_user(request: Request):
    """Redirect DAO users to the DAO cases page if they try to access non-DAO pages."""
    if get_user_role(request) == 'dao':
        return RedirectResponse('/dao/cases', status_code=302)
    return None


#Load .env and session secret
load_dotenv()
SESSION_SECRET = os.environ.get('SESSION_SECRET')
if not SESSION_SECRET:
    import warnings
    warnings.warn('SESSION_SECRET not set; falling back to insecure default. Set SESSION_SECRET in .env')
    SESSION_SECRET = '4z_WvP9_nL6Wz5xR8qK2mJ-V1tY7bN4uX0iE3oP1aQ8'

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET) 

#Scheduler control event for background jobs
scheduler_stop_event = threading.Event()

#Interval (minutes) for SME alert population; configurable via env
SME_POPULATE_INTERVAL_MINUTES = int(os.environ.get('SME_POPULATE_INTERVAL_MINUTES', '60'))
# Lookback hours and probability threshold for SME population (configurable via env)
SME_POPULATE_LOOKBACK_HOURS = int(os.environ.get('SME_POPULATE_LOOKBACK_HOURS', '24'))
SME_POPULATE_THRESHOLD = float(os.environ.get('SME_POPULATE_THRESHOLD', '0.5'))

#Initialize module engines
db_config ={
    'host': 'localhost',
    'user': 'root',
    'password': 'Bant@6963',
    'database': 'lon-default'
}

try:
    etl_engine = ETLEngine(db_config)
    alerts_engine = AlertsEngine(db_config)
    case_management = CaseManagement(db_config)
    model_governance = ModelGovernance(db_config)
    simulation_engine = SimulationEngine(db_config)
    logger.info("All module engines initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize module engines: {e}")
    etl_engine = None
    alerts_engine = None
    case_management = None
    model_governance = None
    simulation_engine = None

# Wire model-governance drift investigations to the alerts engine so that
# significant drift raises an operational alert automatically.
if model_governance and alerts_engine:
    try:
        def _governance_alert_callback(title, context):
            alerts_engine.create_alert({
                'title': title,
                'severity': 'high',
                'category': 'model_governance',
                'description': f"Drift investigation triggered. Features: {context.get('features')}",
                'metadata': context
            })
        model_governance.set_alert_callback(_governance_alert_callback)
    except Exception as e:
        logger.warning(f"Could not register governance alert callback: {e}")

#If set to '1' (default) DB writes and alert creation will be skipped to
# avoid SSL/schema/runtime DB issues during testing. Set to '0' to enable writes.
SKIP_DB_WRITES = os.environ.get('SKIP_DB_WRITES', '1') == '1'

#3. Add Root Route - redirect to dashboard
@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    """Root route - show login page or redirect to dashboard"""
    user = request.session.get('user')
    if not user:
        # Show login page directly instead of redirecting
        tpl = templates.env.get_template('login.html')
        return HTMLResponse(tpl.render({'request': request}))
    return RedirectResponse('/dashboard', status_code=302)

#4. Add Login POST Route
@app.post('/')
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    
    """Handle login form submission"""
    #Prefer DB-backed authentication if users table exists
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT username, password_hash, is_active FROM users WHERE username = %s LIMIT 1", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and user.get('is_active'):
            stored = user.get('password_hash')
            # If password is hashed, verify using pwd_context; otherwise fall back to plaintext compare
            try:
                if stored and pwd_context.verify(password, stored):
                    request.session['user'] = username
                    return RedirectResponse('/dashboard', status_code=302)
            except Exception:
                # If verification fails (e.g., stored is plain text), compare directly
                if stored == password:
                    request.session['user'] = username
                    return RedirectResponse('/dashboard', status_code=302)

        # Fallback: allow local admin/admin for emergency access
        if username == 'admin' and password == 'admin':
            request.session['user'] = username
            return RedirectResponse('/dashboard', status_code=302)

        tpl = templates.env.get_template('login.html')
        return HTMLResponse(tpl.render({'request': request, 'error': 'Invalid username or password'}))
    except Exception as e:
        # On DB errors, fall back to local admin login and log the issue
        logger.error(f"Login DB check failed: {e}")
        if username == 'admin' and password == 'admin':
            request.session['user'] = username
            return RedirectResponse('/dashboard', status_code=302)
        tpl = templates.env.get_template('login.html')
        return HTMLResponse(tpl.render({'request': request, 'error': 'Invalid username or password'}))

#5. Move the Dashboard to /dashboard and keep protection
@app.get('/dashboard', response_class=HTMLResponse)
async def dashboard(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)
    tpl = templates.env.get_template('enhanced_dashboard.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

#4. Update Logout to point back to Root
@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse('/', status_code=302)

#5. Handle legacy /index route - redirect to /
@app.get('/index')
async def index_redirect():
    return RedirectResponse('/', status_code=302)

#Module Routes
@app.get('/alerts', response_class=HTMLResponse)
async def alerts_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('alerts.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/cases', response_class=HTMLResponse)
async def cases_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('cases.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/customers', response_class=HTMLResponse)
async def customers_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)
    tpl = templates.env.get_template('customers.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/predict', response_class=HTMLResponse)
async def predict_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('predict.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/etl', response_class=HTMLResponse)
async def etl_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('etl.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/model-governance', response_class=HTMLResponse)
async def model_governance_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('model_governance.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/simulation', response_class=HTMLResponse)
async def simulation_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('simulation.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

#Report Routes
@app.get('/reports/portfolio', response_class=HTMLResponse)
async def portfolio_report(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('portfolio.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

@app.get('/reports/compliance', response_class=HTMLResponse)
async def compliance_report(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/', status_code=302)

    dao_redirect = _redirect_dao_user(request)
    if dao_redirect:
        return dao_redirect
    
    tpl = templates.env.get_template('compliance.html')
    return HTMLResponse(tpl.render({'request': request, 'user': user}))

#Load trained model & preprocessors
assets = joblib.load("models/loan_prediction_assets_1.pkl")
model = assets['model']
te = assets['target_encoder']
scaler = assets['scaler']
label_encoder = assets['label_encoder']
feature_cols = assets['features']
remaining_cats = assets.get('remaining_cats', [])
ord_enc = assets.get('ordinal_encoder')
PREDICTION_HISTORY = []

#Class probability thresholds to promote non-default classes when probability is meaningful.
#Can be overridden via environment variable `CLASS_THRESHOLDS` as JSON, e.g. '{"NPL":0.01,"SME":0.03}'
try:
    _ct = os.environ.get('CLASS_THRESHOLDS')
    if _ct:
        CLASS_THRESHOLDS = json.loads(_ct)
    else:
        # Default thresholds will be set to last-calibrated values if available;
        # fallback to conservative defaults if file is missing or invalid.
        CLASS_THRESHOLDS = {'NPL': 0.0, 'SME': 0.06, 'SET': 0.03}
except Exception:
    CLASS_THRESHOLDS = {'NPL': 0.0, 'SME': 0.06, 'SET': 0.03}

logger.info(f"CLASS_THRESHOLDS set to: {CLASS_THRESHOLDS}")

#If a fast-calibrated thresholds file exists, load and apply numeric thresholds
cal_file = 'calibrated_thresholds_fast.json'
if os.path.exists(cal_file):
    try:
        with open(cal_file, 'r') as _f:
            _cal = json.load(_f)
        # _cal may contain objects with 'threshold' keys or numeric values
        for _k, _v in _cal.items():
            try:
                if isinstance(_v, dict) and 'threshold' in _v:
                    CLASS_THRESHOLDS[_k] = float(_v['threshold'])
                else:
                    CLASS_THRESHOLDS[_k] = float(_v)
            except Exception:
                # skip invalid entries
                continue
        logger.info(f"Loaded calibrated thresholds from {cal_file}: {CLASS_THRESHOLDS}")
    except Exception as e:
        logger.error(f"Failed to load calibrated thresholds from {cal_file}: {e}")


def prepare_model_input(customer_data):
    """Prepare customer data for model prediction using simplified encoding"""
    
    try:
        #Create a copy to avoid modifying original data
        data = customer_data.copy()
        
        #Map database column names to expected model feature names
        column_mapping = {
            'APPROVED_AMOUNT': 'APPROVED_AMOUNT',
            'PRINCIPAL_OS': 'PRINCIPAL_OS', 
            'INTEREST_OS': 'INTEREST_OS',
            'PRINCIPAL_ARREARS': 'PRINCIPAL_ARREARS',
            'CURRENT_COMMITTMENT': 'CURRENT_COMMITTMENT',
            'INSTALLMENT_AMOUNT': 'INSTALLMENT_AMOUNT',
            'COLLATERAL_VALUE': 'COLLATERAL_VALUE',
            'TENURE': 'TENURE',
            'TERM': 'TERM',
            'LOAN_TYPE': 'LOAN_TYPE',
            'LOAN_DESCRIPTION': 'LOAN_DESCRIPTION',
            'LOAN_PRODUCT': 'LOAN_PRODUCT',
            'LTYPE': 'LTYPE',
            'BRANCHNAME': 'BRANCHNAME',
            'DISTRICTNAME': 'DISTRICTNAME',
            'CBE_REGION': 'CBE_REGION',
            'ECONOMIC_SECTOR': 'ECONOMIC_SECTOR',
            'INDUSTRY': 'INDUSTRY',
            'OWNERSHIP': 'OWNERSHIP',
            'SECTOR': 'SECTOR',
            'TERM_OF_PAYMENT': 'TERM_OF_PAYMENT',
            'PRODUCT_OWNER': 'PRODUCT_OWNER',
            "TOTAL_LOAN_DAYS": "TOTAL_LOAN_DAYS",
            "LOAN_AGE_DAYS": "LOAN_AGE_DAYS"
        }
        
        #Rename columns to match model expectations
        data = data.rename(columns=column_mapping)
    
        #Fill missing categorical values with 'Unknown'
        categorical_cols = ['TENURE', 'TERM', 'LOAN_TYPE', 'LOAN_DESCRIPTION', 'LOAN_PRODUCT', 
                          'ECONOMIC_SECTOR', 'INDUSTRY', 'OWNERSHIP', 'SECTOR', 'TERM_OF_PAYMENT',
                          'PRODUCT_OWNER', 'DISTRICTNAME','CBE_REGION', 
                          'BRANCHNAME', 'LTYPE']
        
        for col in categorical_cols:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        
        #Fill missing numeric values with mean
        numeric_cols = ['APPROVED_AMOUNT', 'PRINCIPAL_OS', 'INTEREST_OS',
                      'PRINCIPAL_ARREARS', 'CURRENT_COMMITTMENT', 'INSTALLMENT_AMOUNT', 
                      'COLLATERAL_VALUE',"TOTAL_LOAN_DAYS", "LOAN_AGE_DAYS"]
        
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].mean() if not data[col].isna().all() else 0)
        
        #Ensure all expected columns are present
        for col in feature_cols:
            if col not in data.columns:
                data[col] = 0
        
        #Select only the features model was trained on
        X = data[feature_cols].copy()

        #Apply trained TargetEncoder for high-cardinality categorical columns if available
        try:
            if 'te' in globals() and te is not None:
                # TargetEncoder.transform returns a DataFrame with encoded columns
                X = te.transform(X)
        except Exception:
            # If target encoder fails, continue with original X
            pass
        
        #Categorical encoding - prefer trained encoders; fall back to a stable hash mapping
        categorical_features = ['TENURE', 'TERM', 'LOAN_TYPE', 'LOAN_DESCRIPTION', 'LOAN_PRODUCT', 
                               'ECONOMIC_SECTOR', 'INDUSTRY', 'OWNERSHIP', 'SECTOR', 'TERM_OF_PAYMENT',
                               'PRODUCT_OWNER', 'DISTRICTNAME','CBE_REGION', 
                               'BRANCHNAME', 'LTYPE']

        #Try to use a fitted ordinal encoder from assets for consistent encodings
        enc_cols = [c for c in categorical_features if c in X.columns]
        if ord_enc is not None and len(enc_cols) > 0:
            try:
                #Prepare a copy of the categorical subset and ensure string type
                X_cat = X[enc_cols].astype(str).fillna('Unknown')
                #OrdinalEncoder expects 2D array
                X_trans = ord_enc.transform(X_cat)
                #Write back transformed columns
                for i, c in enumerate(enc_cols):
                    X[c] = X_trans[:, i]
            except Exception:
                # Fallback to deterministic hash mapping if encoder fails
                for c in enc_cols:
                    X[c] = X[c].fillna('Unknown').astype(str).apply(lambda v: float(abs(hash(v)) % 1000))
        else:
            #No fitted encoder available: map categories to stable integer via deterministic hash
            for c in enc_cols:
                X[c] = X[c].fillna('Unknown').astype(str).apply(lambda v: float(abs(hash(v)) % 1000))
        
        #Finally, apply scaler if available (not required for tree models but matches training pipeline)
        #Ensure all columns are numeric: coerce and replace non-numeric with deterministic hash
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].astype(str)
            # Try numeric conversion
            X[col] = pd.to_numeric(X[col], errors='coerce')
            # Replace remaining NaNs (from non-numeric strings) with deterministic hash values
            if X[col].isna().any():
                X[col] = X[col].fillna(0)
                # For positions where original was non-numeric, map original string to hash
                # Re-extract original strings from the input data copy 'data'
                if col in data.columns:
                    orig = data[col].astype(str).fillna('Unknown')
                    mask = pd.to_numeric(orig, errors='coerce').isna()
                    if mask.any():
                        # compute hash mapping for masked rows and assign
                        hashed = orig[mask].apply(lambda v: float(abs(hash(v)) % 1000))
                        X.loc[mask, col] = hashed

        try:
            if scaler is not None:
                X_array = scaler.transform(X)
            else:
                X_array = X.values.astype(np.float64)
        except Exception:
            # Fallback to raw numpy values on any unexpected error
            X_array = X.values.astype(np.float64)

        return X_array
        
    except Exception as e:
        logger.error(f"Error preparing model input: {e}")
        #Return a basic numeric array as fallback
        return np.array([[0] * len(feature_cols)], dtype=np.float64)


def get_prediction_df():
    if len(PREDICTION_HISTORY) == 0:
        return pd.DataFrame()
    return pd.DataFrame(PREDICTION_HISTORY)


def append_prediction_record(record):
    #store both raw inputs and risk outputs for API analytics endpoints
    PREDICTION_HISTORY.append(record)


def parse_date_safe(s, name):
    try:
        #coerce invalid/ out-of-range dates to NaT instead of raising
        dt = pd.to_datetime(s, errors='coerce')
    except Exception:
        dt = pd.NaT
    if pd.isna(dt):


        raise HTTPException(status_code=400, detail=f"Invalid date for {name}: {s}")
    return dt

@app.get('/api/kpis')
async def get_kpis():
    """Get dashboard KPIs from actual database data"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Bant@6963',
            database='lon-default'
        )
        
        cursor = conn.cursor()
        
        # Use the latest prediction per customer contract to avoid duplicate rows
        cursor.execute('''
            SELECT p.predicted_status, COUNT(*) as count 
            FROM prediction_results p
            JOIN (
                SELECT contract_code, MAX(id) AS max_id
                FROM prediction_results
                GROUP BY contract_code
            ) latest
            ON p.contract_code = latest.contract_code AND p.id = latest.max_id
            GROUP BY p.predicted_status
        ''')
        status_counts = dict(cursor.fetchall())

        # Total portfolio should reflect unique customers in the system
        cursor.execute('SELECT COUNT(*) FROM customers')
        total_portfolio = cursor.fetchone()[0] or 0
        
        # Calculate dashboard metrics from real data
        total_customers = total_portfolio
        pas = status_counts.get('PAS', 0)
        set_count = status_counts.get('SET', 0)
        npl = status_counts.get('NPL', 0)
        sme = status_counts.get('SME', 0)
        
        #Active Loans = customers with PAS and SET status
        active_loans = pas + set_count
        
        #Risk Alerts = Total NPL and SME customers
        risk_alerts = npl + sme
        
        # Calculate realistic change percentages based on actual data
        active_loans_change = round((active_loans / max(total_customers, 1)) * 100, 1) if total_customers > 0 else 0
        risk_alerts_change = round((risk_alerts / max(total_customers, 1)) * 100, 1) if total_customers > 0 else 0
        
        cursor.close()
        conn.close()
        
        return {
            'total_customers': total_customers,
            'total_portfolio': total_portfolio,
            'active_loans': active_loans,
            'risk_alerts': risk_alerts,
            'active_loans_change': active_loans_change,
            'risk_alerts_change': risk_alerts_change
        }
        
    except Exception as e:
        print(f"Error getting KPIs: {e}")
        return {
            'total_customers': 0,
            'active_loans': 0,
            'risk_alerts': 0,
            'total_portfolio': 0,
            'active_loans_change': 0,
            'risk_alerts_change': 0
        }

@app.get('/api/risk-distribution')
async def get_risk_distribution():
    """Get risk distribution by CBE Region for NPL and SME customers from actual database"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Bant@6963',
            database='lon-default'
        )
        
        cursor = conn.cursor()
        
        #Get actual risk distribution by region from database
        cursor.execute('''
            SELECT c.CBE_REGION, p.predicted_status, COUNT(*) as count
            FROM prediction_results p
            JOIN (
                SELECT contract_code, MAX(id) AS max_id
                FROM prediction_results
                GROUP BY contract_code
            ) latest
            ON p.contract_code = latest.contract_code AND p.id = latest.max_id
            JOIN customers c ON p.customer_id = c.id
            WHERE p.predicted_status IN ('NPL', 'SME')
            AND c.CBE_REGION IS NOT NULL
            GROUP BY c.CBE_REGION, p.predicted_status
            ORDER BY c.CBE_REGION, p.predicted_status
        ''')
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not results:
            return {'regions': []}
        
        # Process results into the expected format
        region_data = {}
        for region, status, count in results:
            if region not in region_data:
                region_data[region] = {'npl': 0, 'sme': 0}
            
            if status == 'NPL':
                region_data[region]['npl'] = count
            elif status == 'SME':
                region_data[region]['sme'] = count
        
        # Convert to chart data format
        regions = []
        for region, data in region_data.items():
            total = data['npl'] + data['sme']
            if total > 0:
                regions.append({
                    'region': region,
                    'npl': data['npl'],
                    'sme': data['sme'],
                    'total': total
                })
        
        #Sort by total risk count (descending)
        regions.sort(key=lambda x: x['total'], reverse=True)
        
        return {'regions': regions}
        
    except Exception as e:
        print(f"Error getting risk distribution: {e}")
        return {'regions': []}


@app.get('/api/portfolio-overview')
async def get_portfolio_overview():
    """Get portfolio overview data from actual database"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Bant@6963',
            database='lon-default'
        )
        
        cursor = conn.cursor()
        
        # Use latest contract prediction to build the portfolio overview
        cursor.execute('''
            SELECT p.predicted_status, COUNT(*) as count 
            FROM prediction_results p
            JOIN (
                SELECT contract_code, MAX(id) AS max_id
                FROM prediction_results
                GROUP BY contract_code
            ) latest
            ON p.contract_code = latest.contract_code AND p.id = latest.max_id
            GROUP BY p.predicted_status
            ORDER BY count DESC
        ''')
        status_data = dict(cursor.fetchall())
        
        # Get loan amount distribution by status using latest contract predictions
        cursor.execute('''
            SELECT c.CONTRACT_CODE, p.predicted_status, c.APPROVED_AMOUNT
            FROM prediction_results p
            JOIN (
                SELECT contract_code, MAX(id) AS max_id
                FROM prediction_results
                GROUP BY contract_code
            ) latest
            ON p.contract_code = latest.contract_code AND p.id = latest.max_id
            JOIN customers c ON p.customer_id = c.id
            WHERE c.APPROVED_AMOUNT IS NOT NULL
        ''')
        loan_data = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        #Calculate portfolio metrics
        total_customers = sum(status_data.values())
        
        # Calculate total approved amount by status
        amount_by_status = {}
        for contract_code, status, amount in loan_data:
            if status not in amount_by_status:
                amount_by_status[status] = 0
            amount_by_status[status] += float(amount)
        
        #Prepare data for chart
        portfolio_data = []
        colors = {
            'PAS': 'rgba(34, 197, 94, 0.8)',    # Green
            'NPL': 'rgba(239, 68, 68, 0.8)',     # Red
            'SME': 'rgba(251, 146, 60, 0.8)',    # Orange
            'SET': 'rgba(59, 130, 246, 0.8)'     # Blue
        }
        
        for status, count in status_data.items():
            portfolio_data.append({
                'status': status,
                'count': count,
                'percentage': round((count / total_customers) * 100, 1) if total_customers > 0 else 0,
                'total_amount': amount_by_status.get(status, 0),
                'color': colors.get(status, 'rgba(156, 163, 175, 0.8)')
            })
        
        return {
            'portfolio_data': portfolio_data,
            'total_customers': total_customers,
            'total_amount': sum(amount_by_status.values())
        }
        
    except Exception as e:
        print(f"Error getting portfolio overview: {e}")
        return {
            'portfolio_data': [],
            'total_customers': 0,
            'total_amount': 0
        }


@app.get('/api/predict_proba')
async def predict_proba(limit: int = 50):
    df = get_prediction_df()
    rows = []
    for _, row in df.head(limit).iterrows():
        rows.append({
            'CONTRACT_CODE': row.get('CONTRACT_CODE', None),
            'PRINCIPAL_OS': int(row.get('PRINCIPAL_OS', 0)),
            'CURRENT_STATUS': row.get('STATUS', None),
            'npl_prob': float(row.get('NPL_PROB', 0.0))
        })
    return {'rows': rows}


@app.get('/api/ews')
async def ews_list(threshold: float = 0.4, limit: int = 50):
    """Early Warning Signals: loans currently PAS but with predicted NPL probability above threshold."""
    df = get_prediction_df()
    results = []
    for _, row in df.iterrows():
        if str(row.get('STATUS')).upper() != 'PAS':
            continue
        npl_prob = float(row.get('NPL_PROB', 0.0))

        if npl_prob >= threshold:
            results.append({
                'CONTRACT_CODE': row.get('CONTRACT_CODE'),
                'PRINCIPAL_OS': int(row.get('PRINCIPAL_OS', 0)),
                'npl_prob': round(npl_prob, 4),
                'DISTRICTNAME': row.get('DISTRICTNAME'),
                'ECONOMIC_SECTOR': row.get('ECONOMIC_SECTOR')
            })
        if len(results) >= limit:
            break

    results = sorted(results, key=lambda x: x['npl_prob'], reverse=True)
    return {'ews': results}


@app.get('/api/temporal/vintage')
async def vintage_analysis(limit_months: int = 12):
    df = get_prediction_df()
    if df.empty:
        return {'vintage': []}

    if 'GRANT_DATE' in df.columns:
        dt = pd.to_datetime(df['GRANT_DATE'], errors='coerce')
        df['grant_ym'] = dt.dt.to_period('M').astype(str)
    else:
        df['grant_ym'] = ''

    grouped = df.groupby('grant_ym')
    rows = []
    for ym, g in grouped:
        total = len(g)
        npl = int((g['STATUS'] == 'NPL').sum())
        rows.append({'grant_ym': ym, 'npl_rate': round(100 * npl / total, 2) if total else 0, 'total': int(total)})
    rows = sorted(rows, key=lambda x: x['grant_ym'], reverse=True)[:limit_months]
    return {'vintage': rows}


@app.get('/api/temporal/maturity')
async def maturity_concentration(next_months: int = 12):
    df = get_prediction_df()
    if df.empty:
        return {'maturity': []}

    if 'EXPIRY_DATE' in df.columns:
        dt = pd.to_datetime(df['EXPIRY_DATE'], errors='coerce')
        df['expiry_ym'] = dt.dt.to_period('M').astype(str)
    else:
        df['expiry_ym'] = ''

    grouped = df.groupby('expiry_ym')['PRINCIPAL_OS'].sum().reset_index()
    grouped = grouped.sort_values('expiry_ym')
    rows = [{'expiry_ym': r['expiry_ym'], 'sum_principal': int(r['PRINCIPAL_OS'])} for _, r in grouped.iterrows()]
    return {'maturity': rows}


@app.get('/api/sector_exposure')
async def sector_exposure():
    df = get_prediction_df()
    if df.empty:
        return {'sectors': []}

    grouped = df.groupby('ECONOMIC_SECTOR').agg({'PRINCIPAL_OS': 'sum', 'CONTRACT_CODE': 'count'}).reset_index()
    rows = [{'sector': r['ECONOMIC_SECTOR'], 'sum_principal': int(r['PRINCIPAL_OS']), 'count': int(r['CONTRACT_CODE'])} for _, r in grouped.iterrows()]
    return {'sectors': rows}


@app.get('/api/status_distribution')
async def status_distribution():
    df = get_prediction_df()
    if df.empty:
        return {'status_distribution': []}
    counts = df['STATUS'].value_counts().to_dict()
    rows = [{'status': k, 'count': int(v)} for k, v in counts.items()]
    return {'status_distribution': rows}


@app.get('/api/loan_age_vs_risk')
async def loan_age_vs_risk(limit: int = 500):
    df = get_prediction_df().head(limit)
    if df.empty:
        return {'points': []}
    points = []
    for _, row in df.iterrows():
        points.append({
            'loan_age_days': float(row.get('LOAN_AGE_DAYS', 0)),
            'npl_prob': float(row.get('NPL_PROB', 0.0)),
            'status': row.get('STATUS')
        })
    return {'points': points}

@app.get('/api/explain/{contract_code}')
async def explain_loan(contract_code: str):
    """Lightweight explanation: return top features by (importance * deviation) ranking."""
    df = get_prediction_df()
    row = df.loc[df['CONTRACT_CODE'] == contract_code]
    if row.empty:
        raise HTTPException(status_code=404, detail='Contract not found')
    row = row.iloc[0]

    # Try to provide SHAP explanations if available
    # Prepare single-row features
    X_df = pd.DataFrame([row])
    X = prepare_model_input(X_df)

    if not _SHAP_AVAILABLE:
        # Fallback: return the lightweight heuristic used before
        # Get feature importances if available
        importances = None
        try:
            if hasattr(model, 'feature_importances_'):
                importances = list(model.feature_importances_)
        except Exception:
            importances = None

        feats = feature_cols
        values = X.flatten().tolist() if hasattr(X, 'flatten') else list(X[0])
        contributions = []
        for fname, val, imp in zip(feats, values, importances or [1.0]*len(feats)):
            contributions.append({'feature': fname, 'value': float(val), 'importance': float(imp), 'score': float(abs(val) * imp)})
        contributions = sorted(contributions, key=lambda x: x['score'], reverse=True)[:10]
        return {'contract_code': contract_code, 'top_contributors': contributions, 'shap_available': False}

    #Initialize SHAP explainer lazily
    global _SHAP_EXPLAINER
    try:
        if _SHAP_EXPLAINER is None:
            # Prefer TreeExplainer for tree models
            try:
                _SHAP_EXPLAINER = shap.Explainer(model)
            except Exception:
                try:
                    _SHAP_EXPLAINER = shap.TreeExplainer(model)
                except Exception:
                    sample_df = get_prediction_df().head(50)
                    if sample_df.empty:
                        sample_df = pd.DataFrame([row])
                    _SHAP_EXPLAINER = shap.KernelExplainer(model.predict, prepare_model_input(sample_df))

        #Compute SHAP values for the single row
        shap_vals = _SHAP_EXPLAINER(X)
        #shap_vals may be a Explanation object
        if hasattr(shap_vals, 'values'):
            values = shap_vals.values[0].tolist()
            base_value = float(shap_vals.base_values[0]) if hasattr(shap_vals, 'base_values') else 0.0
        else:
            #fallback
            values = list(shap_vals[0])
            base_value = 0.0

        feats = feature_cols
        contributions = []
        for fname, val in zip(feats, values):
            contributions.append({'feature': fname, 'shap_value': float(val)})
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)[:15]
        return {'contract_code': contract_code, 'base_value': base_value, 'shap_available': True, 'top_contributors': contributions}
    except Exception as exc:
        #If SHAP fails, return fallback explanation and a note
        importances = None
        try:
            if hasattr(model, 'feature_importances_'):
                importances = list(model.feature_importances_)
        except Exception:
            importances = None
        feats = feature_cols
        values = X.flatten().tolist() if hasattr(X, 'flatten') else list(X[0])
        contributions = []
        for fname, val, imp in zip(feats, values, importances or [1.0]*len(feats)):
            contributions.append({'feature': fname, 'value': float(val), 'importance': float(imp), 'score': float(abs(val) * imp)})
        return {'contract_code': contract_code, 'top_contributors': contributions, 'shap_available': False, 'error': str(exc)}


@app.post("/predict")

async def predict(
    CONTRACT_CODE: str = Form(...),
    DISTRICTNAME: str = Form(...),
    CBE_REGION: str = Form(...),
    BRANCHNAME: str = Form(...),
    APPROVED_AMOUNT: float = Form(...),
    TENURE: str = Form(...),
    TERM: str = Form(...),
    LOAN_TYPE: str = Form(...),
    LOAN_DESCRIPTION: str = Form(...),
    LOAN_PRODUCT: str = Form(...),
    LTYPE: str = Form(...),
    PRINCIPAL_OS: float = Form(...),
    INTEREST_OS: float = Form(...),
    PRINCIPAL_ARREARS: float = Form(...),
    CURRENT_COMMITTMENT: float = Form(...),
    INSTALLMENT_AMOUNT: float = Form(...),
    ECONOMIC_SECTOR: str = Form(...),
    INDUSTRY: str = Form(...),
    OWNERSHIP: str = Form(...),
    SECTOR: str = Form(...),
    TERM_OF_PAYMENT: str = Form(...),
    PRODUCT_OWNER: str = Form(...),
    COLLATERAL_VALUE: float = Form(...),
    GRANT_DATE: str = Form(...),
    EXPIRY_DATE: str = Form(...),
    BUSINESS_DATE: str = Form(...),
):

    #Parse dates (use safe parser to return 400 on invalid input)
    grant_dt = parse_date_safe(GRANT_DATE, "GRANT_DATE")
    expiry_dt = parse_date_safe(EXPIRY_DATE, "EXPIRY_DATE")
    business_dt = parse_date_safe(BUSINESS_DATE, "BUSINESS_DATE")

    #Ensure explanation variable always exists to avoid UnboundLocalError later
    explanation = None

    #=== Feature Engineering ===
    total_loan_days = (expiry_dt - grant_dt).days
    loan_age_days = (business_dt - grant_dt).days


    #Build input DataFrame
    data = pd.DataFrame([{
        "DISTRICTNAME": DISTRICTNAME,
        "CBE_REGION": CBE_REGION,
        "BRANCHNAME": BRANCHNAME,
        "APPROVED_AMOUNT": APPROVED_AMOUNT,
        "TENURE": TENURE,
        "TERM": TERM,
        "LOAN_TYPE": LOAN_TYPE,
        "LOAN_DESCRIPTION": LOAN_DESCRIPTION,
        "LOAN_PRODUCT": LOAN_PRODUCT,
        "LTYPE": LTYPE,
        "PRINCIPAL_OS": PRINCIPAL_OS,
        "INTEREST_OS": INTEREST_OS,
        "PRINCIPAL_ARREARS": PRINCIPAL_ARREARS,
        "CURRENT_COMMITTMENT": CURRENT_COMMITTMENT,
        "INSTALLMENT_AMOUNT": INSTALLMENT_AMOUNT,
        "ECONOMIC_SECTOR": ECONOMIC_SECTOR,
        "INDUSTRY": INDUSTRY,
        "OWNERSHIP": OWNERSHIP,
        "SECTOR": SECTOR,
        "TERM_OF_PAYMENT": TERM_OF_PAYMENT,
        "PRODUCT_OWNER": PRODUCT_OWNER,
        "COLLATERAL_VALUE": COLLATERAL_VALUE,
        "TOTAL_LOAN_DAYS": total_loan_days,
        "LOAN_AGE_DAYS": loan_age_days,
    }])

    #Use enhanced prepare_model_input function
    X_array = prepare_model_input(data)
    
    #Enable categorical support for XGBoost
    model.enable_categorical = True

    #Predict
    pred_encoded = model.predict(X_array)[0]
    predicted_status = label_encoder.inverse_transform([pred_encoded])[0]

    #Get all class probabilities
    all_probabilities = {}
    npl_prob = 0.0
    sme_prob = 0.0
    
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_array)
            classes = label_encoder.classes_
            
            for i, class_name in enumerate(classes):
                if i < len(probs[0]):
                    all_probabilities[class_name] = float(probs[0][i])
            
            npl_prob = all_probabilities.get('NPL', 0.0)
            sme_prob = all_probabilities.get('SME', 0.0)
        else:
            # Fallback if no predict_proba
            all_probabilities = {cls: (1.0 if cls == predicted_status else 0.0) for cls in label_encoder.classes_}
            npl_prob = 1.0 if predicted_status == 'NPL' else 0.0
            sme_prob = 1.0 if predicted_status == 'SME' else 0.0
    except Exception as e:
        logger.error(f"Error getting probabilities: {e}")
        all_probabilities = {cls: 0.0 for cls in label_encoder.classes_}
        npl_prob = 0.0
        sme_prob = 0.0

    # Apply class thresholds to possibly override predicted_status (promote NPL/SME/SET)
    try:
        thresholds = CLASS_THRESHOLDS
    except NameError:
        thresholds = {'NPL': 0.20, 'SME': 0.15, 'SET': 0.15}

    # Find non-PAS classes meeting their thresholds
    candidates = [(cls, prob) for cls, prob in all_probabilities.items() if cls != 'PAS' and prob >= thresholds.get(cls, 1.0)]
    if candidates:
        # Pick the non-PAS class with highest probability among those meeting threshold
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = candidates[0][0]
        try:
            logger.info(f"Overriding predicted_status {predicted_status} -> {chosen} due to threshold; probs={all_probabilities}")
        except Exception:
            pass
        predicted_status = chosen

    #Prepare result data
    result_row = data.iloc[0].to_dict()
    result_row.update({
        'STATUS': predicted_status,
        'NPL_PROB': npl_prob,  #Keep for backward compatibility
        'GRANT_DATE': GRANT_DATE,
        'EXPIRY_DATE': EXPIRY_DATE,
        'BUSINESS_DATE': BUSINESS_DATE,
        'CONTRACT_CODE': CONTRACT_CODE
    })

    #Store in customer table and prediction_results
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        #Check if customer exists
        cursor.execute("SELECT CONTRACT_CODE FROM customers WHERE CONTRACT_CODE = %s", (CONTRACT_CODE,))
        existing = cursor.fetchone()
        
        if existing:
            #Update existing customer
            update_fields = [
                'DISTRICTNAME', 'CBE_REGION', 'BRANCHNAME', 'APPROVED_AMOUNT',
                'TENURE', 'TERM', 'LOAN_TYPE', 'LOAN_DESCRIPTION', 'LOAN_PRODUCT', 'LTYPE',
                'PRINCIPAL_OS', 'INTEREST_OS', 'PRINCIPAL_ARREARS', 'CURRENT_COMMITTMENT',
                'INSTALLMENT_AMOUNT', 'ECONOMIC_SECTOR', 'INDUSTRY', 'OWNERSHIP', 'SECTOR',
                'TERM_OF_PAYMENT', 'PRODUCT_OWNER','COLLATERAL_VALUE',
                'UPDATED_AT'
            ]
            
            set_clause = ", ".join([f"{field} = %s" for field in update_fields])
            # Use .get() to avoid KeyError when result_row lacks optional fields
            values = [result_row.get(field, None) for field in update_fields] + [datetime.now()]
            
            cursor.execute(f"""
                UPDATE customers 
                SET {set_clause}
                WHERE CONTRACT_CODE = %s
            """, values + [CONTRACT_CODE])
        else:
            #Insert new customer
            insert_fields = [
                'CONTRACT_CODE', 'DISTRICTNAME', 'CBE_REGION', 'BRANCHNAME',
                'APPROVED_AMOUNT', 'TENURE', 'TERM', 'LOAN_TYPE', 'LOAN_DESCRIPTION',
                'LOAN_PRODUCT', 'LTYPE', 'PRINCIPAL_OS', 'INTEREST_OS', 'PRINCIPAL_ARREARS',
                'CURRENT_COMMITTMENT', 'INSTALLMENT_AMOUNT', 'ECONOMIC_SECTOR', 'INDUSTRY',
                'OWNERSHIP', 'SECTOR', 'TERM_OF_PAYMENT', 'PRODUCT_OWNER',
                'COLLATERAL_VALUE','CREATED_AT', 'UPDATED_AT'
            ]
            
            placeholders = ", ".join(["%s"] * len(insert_fields))
            # Use .get() to avoid KeyError when result_row lacks optional fields
            values = [result_row.get(field, None) for field in insert_fields] + [datetime.now(), datetime.now()]
            
            cursor.execute(f"""
                INSERT INTO customers ({', '.join(insert_fields)})
                VALUES ({placeholders})
            """, values)
        
        conn.commit()
        
        #Store prediction results in prediction_results table
        cursor.execute("""
            INSERT INTO prediction_results 
            (contract_code, predicted_status, npl_probability, pas_probability, sme_probability, set_probability, risk_level, prediction_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            predicted_status = VALUES(predicted_status),
            npl_probability = VALUES(npl_probability),
            pas_probability = VALUES(pas_probability),
            sme_probability = VALUES(sme_probability),
            set_probability = VALUES(set_probability),
            risk_level = VALUES(risk_level),
            prediction_date = VALUES(prediction_date)
        """, (
            CONTRACT_CODE,
            predicted_status,
            all_probabilities.get('NPL', 0.0),
            all_probabilities.get('PAS', 0.0),
            all_probabilities.get('SME', 0.0),
            all_probabilities.get('SET', 0.0),
            'High Risk' if predicted_status == 'NPL' else 'Medium Risk' if predicted_status in ['SME', 'SET'] else 'Low Risk',
            datetime.now()
        ))
        
        conn.commit()
        cursor.close()
        
        #Generate alerts if risk is high
        if alerts_engine and sme_prob > 0.5:
            customer_data = {
                'CONTRACT_CODE': CONTRACT_CODE,
                'SME_PROBABILITY': sme_prob,
                'PRINCIPAL_OS': PRINCIPAL_OS,
                'BRANCHNAME': BRANCHNAME,
                'LOAN_STATUS': predicted_status,
                'BUSINESS_DATE': BUSINESS_DATE,
                'days_past_due': loan_age_days
            }
            
            triggered_alerts = alerts_engine.evaluate_alert_conditions(customer_data)
            for alert in triggered_alerts:
                alerts_engine.create_alert(alert)
        
        #Generate SHAP explanation if model governance is available
        #Keep explanation generation separate and tolerant to errors
        explanation = None
        if model_governance:
            try:
                explanation = model_governance.generate_shap_explanation(result_row)
            except Exception as e:
                logger.error(f"Failed to generate SHAP explanation: {e}")
        
    except Exception as e:
        logger.error(f"Failed to store prediction in database: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

    #Keep backward compatibility with prediction history
    try:
        append_prediction_record(result_row)
    except Exception:
        # never allow history append to break response
        logger.debug("Failed to append prediction record to history, continuing")

    #Determine risk level
    if predicted_status == 'NPL':
        risk_level = 'High Risk'
    elif predicted_status == 'SET':
        risk_level = 'Medium Risk'
    elif predicted_status == 'SME':
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'

    response = {
        "prediction": predicted_status,
        "npl_probability": round(all_probabilities.get('NPL', 0.0), 4),
        "pas_probability": round(all_probabilities.get('PAS', 0.0), 4),
        "sme_probability": round(all_probabilities.get('SME', 0.0), 4),
        "set_probability": round(all_probabilities.get('SET', 0.0), 4),
        "all_probabilities": all_probabilities,
        "risk_level": risk_level,
        "contract_code": CONTRACT_CODE,
        "analysis": {
            "total_loan_days": total_loan_days,
            "loan_age_days": loan_age_days,
            "approved_amount": float(APPROVED_AMOUNT)
        }
    }
    
    #Add explanation if available (safe checks without referencing local var directly)
    try:
        explanation_obj = locals().get('explanation', None)
        if isinstance(explanation_obj, dict):
            if 'error' not in explanation_obj:
                response["explanation"] = explanation_obj
        elif explanation_obj is not None:
            response["explanation"] = explanation_obj
    except Exception:
        logger.debug("Skipping explanation due to unexpected format")
    
    return response

#===== NEW MODULE API ENDPOINTS =====

#ETL Engine Endpoints
@app.get('/api/etl/quality-dashboard')
async def etl_quality_dashboard():
    """Get ETL data quality dashboard"""
    if not etl_engine:
        return {'error': 'ETL engine not available'}
    return etl_engine.get_data_quality_dashboard()

@app.post('/api/etl/process-file')
async def etl_process_file(file_path: str = Form(...)):
    """Process a file through ETL pipeline"""
    if not etl_engine:
        return {'error': 'ETL engine not available'}
    return etl_engine.process_etl_pipeline(file_path)

#Alerts Engine Endpoints
@app.get('/api/alerts/dashboard')
async def alerts_dashboard():
    """Get alerts dashboard"""
    try:
        if alerts_engine:
            dashboard_data = alerts_engine.get_alerts_dashboard()
            # Attach scheduler/config values so frontend can show current settings
            dashboard_data['sme_scheduler'] = {
                'interval_minutes': SME_POPULATE_INTERVAL_MINUTES,
                'lookback_hours': SME_POPULATE_LOOKBACK_HOURS,
                'threshold': SME_POPULATE_THRESHOLD
            }
            return dashboard_data
        else:
            return {'error': 'Alerts engine not available'}
    except Exception as e:
        logger.error(f"Failed to get alerts dashboard: {e}")
        return {'error': str(e)}


def _sme_populate_loop(interval_minutes: int = 60, lookback_hours: int = 24, threshold: float = 0.5):
    """Daemon thread loop to periodically populate SME alerts.

    interval_minutes: how often to run
    lookback_hours: how far back to look at predictions
    threshold: SME probability threshold
    """
    logger.info(f"SME populate scheduler starting (interval {interval_minutes} minutes, lookback {lookback_hours}h, threshold {threshold})")
    try:
        while not scheduler_stop_event.is_set():
            try:
                if alerts_engine:
                    logger.info('Running scheduled SME alert population')
                    alerts_engine.populate_sme_alerts(since_hours=lookback_hours, threshold=threshold)
                else:
                    logger.warning('Alerts engine not initialized; skipping scheduled SME populate')
            except Exception as e:
                logger.error(f"Scheduled SME populate failed: {e}")

            #Wait with early exit support
            scheduler_stop_event.wait(interval_minutes * 60)
    finally:
        logger.info('SME populate scheduler stopped')


def _model_eval_loop(interval_minutes: int = 360):
    """Daemon thread loop to periodically run the model governance monitoring cycle.

    Evaluates predictions against actual outcomes, detects drift (KS test), and
    triggers automated investigations when significant drift is found.
    """
    logger.info(f"Model evaluation scheduler starting (interval {interval_minutes} minutes)")
    try:
        while not scheduler_stop_event.is_set():
            try:
                if model_governance:
                    logger.info('Running scheduled model governance monitoring cycle')
                    result = model_governance.run_monitoring_cycle()
                    perf = result.get('performance') or {}
                    if perf.get('evaluated'):
                        logger.info(
                            f"Model evaluation logged: {perf.get('evaluation_count')} samples, "
                            f"accuracy={perf.get('accuracy')}, f1={perf.get('f1_score')}"
                        )
                    else:
                        logger.info(f"Model evaluation skipped: {perf.get('reason')}")
                    drift = result.get('drift') or {}
                    if drift.get('baseline_captured'):
                        logger.info('Model drift baseline captured')
                    elif drift.get('features_with_drift'):
                        logger.warning(f"Model drift detected on: {drift.get('features_with_drift')}")
                    if result.get('investigation'):
                        logger.warning('Automated model investigation triggered')
                else:
                    logger.warning('Model governance engine not initialized; skipping scheduled evaluation')
            except Exception as e:
                logger.error(f"Scheduled model evaluation failed: {e}")

            # Wait with early exit support
            scheduler_stop_event.wait(interval_minutes * 60)
    finally:
        logger.info('Model evaluation scheduler stopped')


@app.on_event('startup')
def _start_background_jobs():
    # Start SME populate scheduler thread with configured lookback and threshold
    t = threading.Thread(
        target=_sme_populate_loop,
        args=(SME_POPULATE_INTERVAL_MINUTES, SME_POPULATE_LOOKBACK_HOURS, SME_POPULATE_THRESHOLD),
        daemon=True
    )
    t.start()
    logger.info('Background SME populate thread started')


@app.on_event('shutdown')
def _stop_background_jobs():
    scheduler_stop_event.set()
    logger.info('Shutdown signal set for background jobs')

@app.post('/api/alerts/rules')
async def create_alert_rule(rule_data: dict):
    """Create a new alert rule"""
    if not alerts_engine:
        return {'error': 'Alerts engine not available'}
    return alerts_engine.create_alert_rule(rule_data)

# Note: acknowledge_alert and resolve_alert endpoints are defined later in the file with proper JSON handling and auth checks
# Lines moved to around line 1895 to use Request object and JSON parsing

#Alert escalation check endpoint
@app.get('/api/alerts/check-escalations')
async def check_alert_escalations():
    """Check for alerts that need escalation"""
    if not alerts_engine:
        return {'error': 'Alerts engine not available'}
    escalated_count = alerts_engine.check_alert_escalations()
    return {'escalated_count': escalated_count}

@app.post('/api/alerts/populate-sme')
async def api_populate_sme_alerts(since_hours: int = 24, threshold: float = 0.5):
    """API wrapper to populate SME alerts from recent prediction results."""
    if not alerts_engine:
        return {'success': False, 'error': 'Alerts engine not available'}

    try:
        # create default SME rules if missing
        try:
            alerts_engine.create_default_sme_alert_rules()
        except Exception:
            pass

        result = alerts_engine.populate_sme_alerts(since_hours=since_hours, threshold=threshold)
        return result
    except Exception as e:
        logger.error(f"API populate-sme failed: {e}")
        return {'success': False, 'error': str(e)}

#Case Management Endpoints
@app.get('/api/cases/dashboard')
async def cases_dashboard():
    """Get cases dashboard"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    return case_management.get_cases_dashboard()
    
    return case_management.get_my_cases(user_id, status)

@app.post('/api/cases')
async def create_case(case_data: dict):
    """Create a new case"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    return case_management.create_case(case_data)

@app.get('/api/cases/{case_id}')
async def get_case_details(case_id: str):
    """Get case details"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    return case_management.get_case_details(case_id)

@app.post('/api/cases/{case_id}/assign')
async def assign_case(case_id: str, assigned_to: str = Form(...), assigned_by: str = Form(...)):
    """Assign a case to an officer"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    return case_management.assign_case(case_id, assigned_to, assigned_by)

@app.post('/api/cases/{case_id}/activities')
async def add_case_activity(case_id: str, activity_data: dict, user_id: str = Form(...)):
    """Add activity to a case"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    return case_management.add_case_activity(case_id, activity_data, user_id)

@app.post('/api/cases/{case_id}/resolve')
async def resolve_case(case_id: str, resolution_data: dict, user_id: str = Form(...)):
    """Resolve a case"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    return case_management.resolve_case(case_id, resolution_data, user_id)

@app.post('/api/cases/auto-create-from-alerts')
async def auto_create_cases():
    """Auto-create cases from high-priority alerts"""
    if not case_management:
        return {'error': 'Case management engine not available'}
    created_count = case_management.auto_create_cases_from_alerts()
    return {'created_cases_count': created_count}

#Model Governance Endpoints
@app.get('/api/model-governance/dashboard')
async def model_governance_dashboard():
    """Get model governance dashboard"""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    return model_governance.get_model_governance_dashboard()

@app.get('/api/model-governance/explain/{contract_code}')
async def explain_model_prediction(contract_code: str):
    """Get model explanation for a contract"""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    
    #Get customer data from database
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM customers WHERE CONTRACT_CODE = %s", (contract_code,))
        customer = cursor.fetchone()
        cursor.close()
        
        if not customer:
            return {'error': 'Customer not found'}
        
        explanation = model_governance.generate_shap_explanation(customer)
        return explanation
        
    except Exception as e:
        return {'error': str(e)}
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

@app.get('/api/model-governance/detect-drift')
async def detect_data_drift():                                      
    """Detect data drift"""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    return model_governance.detect_data_drift()

@app.post('/api/model-governance/log-performance')
async def log_model_performance(evaluation_date: str = Form(...), metrics: dict = Form(...)):
    """Log model performance metrics"""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    
    try:
        eval_date = datetime.fromisoformat(evaluation_date)
        model_governance.log_model_performance(eval_date, metrics)
        return {'success': True}
    except Exception as e:
        return {'error': str(e)}

@app.post('/api/model-governance/feedback')
async def submit_explanation_feedback(contract_code: str = Form(...), helpful: bool = Form(...), comment: str = Form(None)):
    """Capture user feedback on an explanation to refine the XAI interface."""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    return model_governance.log_explanation_feedback(contract_code, helpful, comment)

@app.get('/api/model-governance/feedback-summary')
async def explanation_feedback_summary():
    """Aggregate explanation feedback for the XAI interface."""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    return model_governance.get_feedback_summary()

@app.get('/api/model-governance/curves')
async def get_performance_curves():
    """Get the latest ROC and Precision-Recall curves for the model."""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    dash = model_governance.get_model_governance_dashboard()
    return {
        'roc': (dash.get('latest_curves') or {}).get('roc'),
        'pr': (dash.get('latest_curves') or {}).get('pr'),
        'latest_evaluation': dash.get('latest_evaluation')
    }

@app.post('/api/model-governance/capture-reference')
async def capture_drift_reference_endpoint():
    """Snap the current population as the drift reference baseline."""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    return model_governance.capture_drift_reference()

@app.post('/api/model-governance/monitor')
async def run_monitoring_cycle_endpoint():
    """Run the full monitoring cycle (evaluate + drift + investigation) on demand."""
    if not model_governance:
        return {'error': 'Model governance engine not available'}
    return model_governance.run_monitoring_cycle()

#Simulation Engine Endpoints
@app.get('/api/simulation/dashboard')
async def simulation_dashboard():
    """Get simulation dashboard"""
    if not simulation_engine:
        return {'error': 'Simulation engine not available'}
    return simulation_engine.get_simulation_dashboard()

@app.post('/api/simulation/scenarios')
async def create_simulation_scenario(scenario_data: dict):
    """Create a new simulation scenario"""
    if not simulation_engine:
        return {'error': 'Simulation engine not available'}
    return simulation_engine.create_simulation_scenario(scenario_data)

@app.post('/api/simulation/{scenario_id}/run')
async def run_simulation(scenario_id: str):
    """Run a simulation scenario"""
    if not simulation_engine:
        return {'error': 'Simulation engine not available'}
    return simulation_engine.run_simulation(scenario_id)

@app.get('/api/simulation/{scenario_id}/results')
async def get_simulation_results(scenario_id: str):
    """Get simulation results"""
    if not simulation_engine:
        return {'error': 'Simulation engine not available'}
    return simulation_engine.get_simulation_results(scenario_id)

@app.post('/api/simulation/compare')
async def compare_scenarios(scenario_ids: List[str]):
    """Compare multiple simulation scenarios"""
    if not simulation_engine:
        return {'error': 'Simulation engine not available'}
    return simulation_engine.compare_scenarios(scenario_ids)

@app.post('/api/simulation/create-stress-tests')
async def create_stress_test_scenarios():
    """Create predefined stress test scenarios"""
    if not simulation_engine:
        return {'error': 'Simulation engine not available'}
    created_scenarios = simulation_engine.create_stress_test_scenarios()
    return {'created_scenarios': created_scenarios}

#Enhanced Dashboard API (integrating all modules)
@app.get('/api/dashboard/comprehensive')
async def comprehensive_dashboard():
    """Get comprehensive dashboard data from all modules"""
    dashboard_data = {
        'timestamp': datetime.now().isoformat(),
        'modules': {}
    }
    
    #Get data from each module
    if etl_engine:
        dashboard_data['modules']['etl'] = etl_engine.get_data_quality_dashboard()
    
    if alerts_engine:
        dashboard_data['modules']['alerts'] = alerts_engine.get_alerts_dashboard()
    
    if case_management:
        dashboard_data['modules']['cases'] = case_management.get_cases_dashboard()
    
    if model_governance:
        dashboard_data['modules']['model_governance'] = model_governance.get_model_governance_dashboard()
    
    if simulation_engine:
        dashboard_data['modules']['simulation'] = simulation_engine.get_simulation_dashboard()
    
    #Get enhanced KPIs from customer table
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_loans,
                SUM(PRINCIPAL_OS) as total_portfolio_value,
                0 as npl_count,
                0 as pas_count,
                0.0 as avg_npl_probability,
                0.0 as npl_exposure
            FROM customers 
        """)
        
        portfolio_kpis = cursor.fetchone()
        dashboard_data['portfolio_kpis'] = portfolio_kpis
        
        cursor.close()
        
    except Exception as e:
        logger.error(f"Failed to get portfolio KPIs: {e}")
        dashboard_data['portfolio_kpis'] = {}
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
    
    return dashboard_data

#Customer Management APIs
@app.post('/api/customers')
async def create_customer(request: Request):
    """Create a new customer"""
    try:
        data = await request.json()
        
        #Debug: Log received data
        logger.info(f"Customer creation - Received data keys: {list(data.keys())}")
        logger.info(f"Customer creation - Received data sample: {dict(list(data.items())[:5])}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        #Insert new customer with simplified schema fields
        sql = """
        INSERT INTO customers (
            CONTRACT_CODE, DISTRICTNAME, CBE_REGION, BRANCHNAME, APPROVED_AMOUNT, 
            GRANT_DATE, EXPIRY_DATE, TENURE, TERM, LOAN_TYPE, LOAN_DESCRIPTION, 
            LOAN_PRODUCT, BUSINESS_DATE, PRINCIPAL_OS, INTEREST_OS, PRINCIPAL_ARREARS, 
            CURRENT_COMMITTMENT, INSTALLMENT_AMOUNT, ECONOMIC_SECTOR, INDUSTRY, 
            OWNERSHIP, SECTOR, TERM_OF_PAYMENT, PRODUCT_OWNER, LTYPE, COLLATERAL_VALUE, 
            CREATED_AT, UPDATED_AT
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(sql, (
            data.get('CONTRACT_CODE'),
            data.get('DISTRICTNAME'),
            data.get('CBE_REGION'),
            data.get('BRANCHNAME'),
            data.get('APPROVED_AMOUNT'),
            data.get('GRANT_DATE'),
            data.get('EXPIRY_DATE'),
            data.get('TENURE'),
            data.get('TERM'),
            data.get('LOAN_TYPE'),
            data.get('LOAN_DESCRIPTION'),
            data.get('LOAN_PRODUCT'),
            data.get('BUSINESS_DATE'),
            data.get('PRINCIPAL_OS'),
            data.get('INTEREST_OS'),
            data.get('PRINCIPAL_ARREARS'),
            data.get('CURRENT_COMMITTMENT'),
            data.get('INSTALLMENT_AMOUNT'),
            data.get('ECONOMIC_SECTOR'),
            data.get('INDUSTRY'),
            data.get('OWNERSHIP'),
            data.get('SECTOR'),
            data.get('TERM_OF_PAYMENT'),
            data.get('PRODUCT_OWNER'),
            data.get('LTYPE'),
            data.get('COLLATERAL_VALUE'),
            datetime.now(),
            datetime.now()
        ))
        
        conn.commit()
        customer_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return {'success': True, 'customer_id': customer_id, 'message': 'Customer created successfully'}
        
    except Exception as e:
        logger.error(f"Failed to create customer: {e}")
        return {'success': False, 'error': str(e)}

@app.get('/api/customers')
async def get_customers():
    """Get all customers"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT c.*, 
                   COALESCE(pr.predicted_status, 'Not Predicted') as predicted_status,
                   COALESCE(pr.risk_level, 'Not Predicted') as risk_level,
                   COALESCE(pr.npl_probability, 0.0) as npl_probability,
                   COALESCE(pr.pas_probability, 0.0) as pas_probability,
                   COALESCE(pr.sme_probability, 0.0) as sme_probability,
                   COALESCE(pr.set_probability, 0.0) as set_probability,
                   pr.prediction_date
            FROM customers c
            LEFT JOIN (
                SELECT DISTINCT pr1.* 
                FROM prediction_results pr1
                INNER JOIN (
                    SELECT customer_id, MAX(prediction_date) as max_date
                    FROM prediction_results
                    GROUP BY customer_id
                ) pr2 ON pr1.customer_id = pr2.customer_id AND pr1.prediction_date = pr2.max_date
            ) pr ON c.id = pr.customer_id
            ORDER BY c.CREATED_AT DESC
        """)
        
        customers = cursor.fetchall()
        
        #Debug: Log sample customer data
        if customers:
            logger.info(f"Sample customer data: {customers[0]}")
            logger.info(f"Available fields: {list(customers[0].keys())}")
        
        cursor.close()
        conn.close()
        
        return {'success': True, 'customers': customers}
        
    except Exception as e:
        logger.error(f"Failed to get customers: {e}")
        return {'success': False, 'error': str(e)}

@app.put('/api/customers/{customer_id}')
async def update_customer(customer_id: int, request: Request):
    """Update a customer"""
    try:
        data = await request.json()
        
        # Debug: Log received data
        logger.info(f"Customer update {customer_id} - Received data keys: {list(data.keys())}")
        logger.info(f"Customer update {customer_id} - Received data sample: {dict(list(data.items())[:5])}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        #Build dynamic update query
        update_fields = []
        values = []
        
        # All possible database fields that can be updated (simplified schema)
        all_fields = [
            'DISTRICTNAME', 'CBE_REGION', 'BRANCHNAME', 'APPROVED_AMOUNT', 
            'GRANT_DATE', 'EXPIRY_DATE', 'TENURE', 'TERM', 'LOAN_TYPE', 
            'LOAN_DESCRIPTION', 'LOAN_PRODUCT', 'BUSINESS_DATE', 'PRINCIPAL_OS', 
            'INTEREST_OS', 'PRINCIPAL_ARREARS', 'CURRENT_COMMITTMENT', 
            'INSTALLMENT_AMOUNT', 'ECONOMIC_SECTOR', 'INDUSTRY', 'OWNERSHIP', 
            'SECTOR', 'TERM_OF_PAYMENT', 'PRODUCT_OWNER', 'LTYPE', 'COLLATERAL_VALUE'
        ]
        
        for field in all_fields:
            if field in data:
                update_fields.append(f"{field} = %s")
                values.append(data[field])
        
        #Debug: Log which fields are being updated
        logger.info(f"Customer update {customer_id} - Fields to update: {update_fields}")
        logger.info(f"Customer update {customer_id} - Number of fields: {len(update_fields)}")
        
        if update_fields:
            sql = f"""
            UPDATE customers 
            SET {', '.join(update_fields)}, UPDATED_AT = NOW()
            WHERE id = %s
            """
            values.append(customer_id)
            
            cursor.execute(sql, values)
            conn.commit()
        
        cursor.close()
        conn.close()
        
        return {'success': True, 'message': 'Customer updated successfully'}
        
    except Exception as e:
        logger.error(f"Failed to update customer: {e}")
        return {'success': False, 'error': str(e)}

@app.delete('/api/customers/{customer_id}')
async def delete_customer(customer_id: int):
    """Delete a customer"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM customers WHERE id = %s", (customer_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {'success': True, 'message': 'Customer deleted successfully'}
        
    except Exception as e:
        logger.error(f"Failed to delete customer: {e}")
        return {'success': False, 'error': str(e)}

@app.post('/api/customers/{customer_id}/predict')
async def predict_customer_risk(customer_id: int):
    """Predict risk for a specific customer"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM customers WHERE id = %s", (customer_id,))
        customer = cursor.fetchone()
        
        if not customer:
            return {'success': False, 'error': 'Customer not found'}
        
        cursor.close()
        conn.close()
        
        #Prepare data for prediction
        customer_data = pd.DataFrame([customer])
        
        #Calculate derived features
        if 'GRANT_DATE' in customer_data.columns and 'EXPIRY_DATE' in customer_data.columns  and 'BUSINESS_DATE' in customer_data.columns:
            grant_date = pd.to_datetime(customer_data['GRANT_DATE'], errors='coerce')
            expiry_date = pd.to_datetime(customer_data['EXPIRY_DATE'], errors='coerce')
            business_date = pd.to_datetime(customer_data['BUSINESS_DATE'], errors='coerce') 
            
            customer_data['TOTAL_LOAN_DAYS'] = (expiry_date - grant_date).dt.days
            customer_data['LOAN_AGE_DAYS'] = (business_date  - grant_date).dt.days
           
        
        #Prepare model input
        X = prepare_model_input(customer_data)
        
        logger.info(f"Making prediction with input shape: {X.shape}")
        #X is a numpy array, so we can't access .columns
        #Use feature_cols for logging instead
        logger.info(f"Input features: {list(feature_cols)}")
        
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        logger.info(f"Raw prediction: {prediction}")
        logger.info(f"Raw prediction probabilities: {prediction_proba}")
        logger.info(f"Prediction type: {type(prediction)}")
        logger.info(f"Probability type: {type(prediction_proba)}")
        
        #Ensure float64 type for consistency
        if hasattr(prediction_proba, 'dtype') and prediction_proba.dtype == np.float32:
            prediction_proba = prediction_proba.astype(np.float64)
        
        #Get class labels
        if hasattr(label_encoder, 'classes_'):
            classes = label_encoder.classes_
            pred_label = classes[prediction]
            
            logger.info(f"Available classes: {classes}")
            logger.info(f"Prediction index: {prediction}")
            logger.info(f"Prediction probabilities: {prediction_proba}")
            
            #Handle multiclass probability extraction
            if len(classes) == 2:
                #Binary case
                npl_prob = prediction_proba[1]
            else:
                #Multiclass case - find NPL probability
                npl_class_index = list(classes).index('NPL') if 'NPL' in classes else 0
                npl_prob = prediction_proba[npl_class_index] if len(prediction_proba) > npl_class_index else 0.0
        
        #Determine risk level based on prediction
        if pred_label == 'NPL':
            risk_level = 'High Risk'
        elif pred_label == 'SME':
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'
        
        logger.info(f"Final prediction result: pred_label={pred_label}, npl_prob={npl_prob}, risk_level={risk_level}")
        
        #Create all class probabilities dictionary
        all_probabilities = {}
        if hasattr(label_encoder, 'classes_'):
            classes = label_encoder.classes_
            for i, class_name in enumerate(classes):
                if i < len(prediction_proba):
                    all_probabilities[class_name] = round(float(prediction_proba[i]), 4)
        else:
            #Fallback for binary case
            all_probabilities['PAS'] = round(float(prediction_proba[0]), 4) if len(prediction_proba) > 0 else 0.0
            all_probabilities['NPL'] = round(float(prediction_proba[1]), 4) if len(prediction_proba) > 1 else 0.0
        
        logger.info(f"All class probabilities: {all_probabilities}")
        
        #Store prediction results in prediction_results table
        conn = get_db_connection()
        cursor = conn.cursor()
        
        #Insert prediction results
        insert_sql = """
        INSERT INTO prediction_results (
            customer_id, contract_code, predicted_status, npl_probability,
            pas_probability, sme_probability, set_probability, risk_level,
            model_version, feature_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        insert_params = (
            customer_id,
            customer.get('CONTRACT_CODE'),
            pred_label,
            npl_prob,
            all_probabilities.get('PAS', 0.0),
            all_probabilities.get('SME', 0.0),
            all_probabilities.get('SET', 0.0),
            risk_level,
            'v1.0',
            31
        )
        
        cursor.execute(insert_sql, insert_params)
        conn.commit()
        logger.info(f"Prediction results stored in prediction_results table")
        
        cursor.close()
        conn.close()
        
        #Generate SME alert if prediction is SME
        if pred_label == 'SME' and alerts_engine:
            try:
                logger.info(f"Generating SME alert for customer {customer_id}")
                #Prepare customer data for alert evaluation
                customer_with_prediction = customer.copy()
                customer_with_prediction['PREDICTED_STATUS'] = pred_label
                
                #Evaluate alert conditions
                triggered_alerts = alerts_engine.evaluate_alert_conditions(customer_with_prediction)
                
                #Create alerts for triggered conditions
                for alert_data in triggered_alerts:
                    alert_result = alerts_engine.create_alert(alert_data)
                    if alert_result.get('success'):
                        logger.info(f"SME alert created: {alert_result.get('alert_id')}")
                    else:
                        logger.error(f"Failed to create SME alert: {alert_result.get('error')}")
                        
            except Exception as e:
                logger.error(f"Failed to generate SME alert: {e}")
        
        result = {
            'success': True,
            'prediction': pred_label,
            'npl_probability': round(npl_prob, 4),
            'all_probabilities': all_probabilities,
            'risk_level': risk_level
        }
        
        logger.info(f"Returning prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to predict customer risk: {e}")
        return {'success': False, 'error': str(e)}

@app.post('/api/alerts/{alert_id}/acknowledge')
async def acknowledge_alert(alert_id: str, request: Request):
    # """Acknowledge an alert"""
    try:
        data = await request.json()
        # Prefer authenticated session user
        session_user = request.session.get('user')
        user_id = session_user or data.get('user_id', 'current_user')

        # Authorization: allow authenticated users or admin fallback
        if not session_user:
            return {'success': False, 'error': 'Forbidden: user not authenticated'}
        
        # Allow admin user (fallback) or users with proper role
        allowed = (session_user == 'admin') or _require_roles(request, ['admin', 'risk_officer', 'branch_manager', 'recovery_officer'])
        if not allowed:
            return {'success': False, 'error': 'Forbidden: insufficient privileges'}

        if alerts_engine:
            result = alerts_engine.acknowledge_alert(alert_id, user_id)
            return result
        else:
            return {'success': False, 'error': 'Alerts engine not initialized'}
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        return {'success': False, 'error': str(e)}

@app.post('/api/alerts/{alert_id}/resolve')
async def resolve_alert(alert_id: str, request: Request):
    """Resolve an alert"""
    try:
        data = await request.json()
        session_user = request.session.get('user')
        user_id = session_user or data.get('user_id', 'current_user')
        resolution_notes = data.get('resolution_notes', '')

        # Authorization: allow authenticated users or admin fallback
        if not session_user:
            return {'success': False, 'error': 'Forbidden: user not authenticated'}
        
        # Allow admin user (fallback) or users with proper role
        allowed = (session_user == 'admin') or _require_roles(request, ['admin', 'risk_officer', 'branch_manager', 'recovery_officer'])
        if not allowed:
            return {'success': False, 'error': 'Forbidden: insufficient privileges'}

        if alerts_engine:
            result = alerts_engine.resolve_alert(alert_id, user_id, resolution_notes)
            return result
        else:
            return {'success': False, 'error': 'Alerts engine not initialized'}
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        return {'success': False, 'error': str(e)}

@app.post('/api/alerts/setup-sme-rules')
async def setup_sme_alert_rules():
    """Create default SME alert rules"""
    try:
        if alerts_engine:
            result = alerts_engine.create_default_sme_alert_rules()
            return result
        else:
            return {'success': False, 'error': 'Alerts engine not initialized'}
    except Exception as e:
        logger.error(f"Failed to setup SME alert rules: {e}")
        return {'success': False, 'error': str(e)}


@app.post('/api/alerts/{alert_id}/escalate')
async def escalate_alert_api(alert_id: str, request: Request):
    """Escalate an alert manually via API"""
    try:
        data = await request.json()
        session_user = request.session.get('user')
        user_id = session_user or data.get('user_id', 'current_user')

        # Authorization: allow authenticated users or admin fallback
        if not session_user:
            return {'success': False, 'error': 'Forbidden: user not authenticated'}
        
        # Allow admin user (fallback) or users with proper role
        allowed = (session_user == 'admin') or _require_roles(request, ['admin', 'risk_officer', 'branch_manager'])
        if not allowed:
            return {'success': False, 'error': 'Forbidden: insufficient privileges'}

        if alerts_engine:
            result = alerts_engine.escalate_alert(alert_id, escalated_by=user_id)
            return result
        else:
            return {'success': False, 'error': 'Alerts engine not initialized'}
    except Exception as e:
        logger.error(f"Failed to escalate alert: {e}")
        return {'success': False, 'error': str(e)}


@app.get('/dao/cases')
async def dao_cases_page(request: Request):
    """Render DAO Cases Management dashboard. Accessible to users with role 'dao' only."""
    session_user = request.session.get('user')
    if not session_user:
        return RedirectResponse('/', status_code=302)

    user_info = _require_roles(request, ['dao'])
    if not user_info:
        return HTMLResponse('Forbidden', status_code=403)

    tpl = templates.env.get_template('dao_cases.html')
    return HTMLResponse(tpl.render({'request': request, 'user': session_user}))


@app.get('/api/dao/cases')
async def api_dao_cases(request: Request):
    """Return cases for DAO to manage (escalated/open/in_progress)."""
    user_info = _require_roles(request, ['dao'])
    if not user_info:
        return {'success': False, 'error': 'Forbidden'}

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT case_id, contract_code, case_type, priority, status, assigned_to, assigned_at, due_date, created_at, resolution_notes
            FROM cases
            WHERE status IN ('escalated','open','in_progress')
            ORDER BY created_at DESC
            LIMIT 1000
            """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {'success': True, 'cases': rows}
    except Exception as e:
        logger.error(f"Failed to fetch DAO cases: {e}")
        return {'success': False, 'error': str(e)}


@app.post('/api/dao/cases/{case_id}/update')
async def api_dao_update_case(case_id: str, request: Request):
    """Update case status or assignment (DAO only). Expects JSON {status, assigned_to} """
    user_info = _require_roles(request, ['dao'])
    if not user_info:
        return {'success': False, 'error': 'Forbidden'}

    try:
        data = await request.json()
        status = data.get('status')
        assigned_to = data.get('assigned_to')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE cases
            SET status = %s, assigned_to = %s, assigned_at = NOW(), updated_at = NOW()
            WHERE case_id = %s
            """,
            (status, assigned_to, case_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {'success': True}
    except Exception as e:
        logger.error(f"Failed to update case {case_id}: {e}")
        return {'success': False, 'error': str(e)}


@app.get('/api/model-governance/fairness')
async def model_fairness(sensitive_attribute: str = 'OWNERSHIP'):
    """Compute fairness metrics for a sensitive attribute"""
    try:
        if model_governance:
            result = model_governance.compute_fairness_metrics(sensitive_attribute)
            return result
        else:
            return {'success': False, 'error': 'Model governance not initialized'}
    except Exception as e:
        logger.error(f"Failed to compute fairness metrics: {e}")
        return {'success': False, 'error': str(e)}


@app.post('/api/cases/{case_id}/remediate')
async def remediate_case(case_id: str, request: Request):
    """Log a remediation action for a case"""
    try:
        data = await request.json()
        # Prefer authenticated session identity
        session_user = request.session.get('user')
        user_id = session_user or data.get('user_id', None)

        # Authorization: only recovery_officer, branch_manager, risk_officer, admin
        allowed = _require_roles(request, ['admin', 'risk_officer', 'branch_manager', 'recovery_officer'])
        if not allowed and user_id != 'system':
            return {'success': False, 'error': 'Forbidden: insufficient privileges'}

        if case_management:
            result = case_management.log_remediation_action(case_id, data, user_id or 'system')
            return result
        else:
            return {'success': False, 'error': 'Case management not initialized'}
    except Exception as e:
        logger.error(f"Failed to log remediation action: {e}")
        return {'success': False, 'error': str(e)}