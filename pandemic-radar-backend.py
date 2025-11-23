
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from celery import Celery
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.types import TypeDecorator, LargeBinary
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from functools import wraps
import os
import base64
import json


# APP CONFIGURATION

app = Flask(__name__)
app.config.update(
    SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', 'postgresql://localhost/pandemic_radar'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    CELERY_BROKER_URL=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    ENCRYPTION_KEY=os.getenv('ENCRYPTION_KEY', Fernet.generate_key()),
    # JWT Configuration
    JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY', 'change-me-in-production'),
    JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=1),
    JWT_REFRESH_TOKEN_EXPIRES=timedelta(days=30),
    JWT_TOKEN_LOCATION=['headers'],
    JWT_HEADER_NAME='Authorization',
    JWT_HEADER_TYPE='Bearer'
)

db = SQLAlchemy(app)
jwt = JWTManager(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["100/hour"])

# Store for revoked tokens (use Redis in production)
revoked_tokens = set()

# Celery setup
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# JWT CALLBACKS AND ROLE-BASED ACCESS

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload['jti']
    return jti in revoked_tokens

@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has been revoked'}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token required'}), 401

def role_required(*allowed_roles):
    """Decorator for role-based access control."""
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            claims = get_jwt()
            user_role = claims.get('role', 'viewer')
            if user_role not in allowed_roles:
                return jsonify({'error': f'Access denied. Required roles: {allowed_roles}'}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# USER MODEL FOR AUTHENTICATION

class User(db.Model):
    """User model for JWT authentication."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default='viewer')  # admin, analyst, responder, viewer
    organization = db.Column(db.String(200))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id, 'email': self.email, 'role': self.role,
            'organization': self.organization, 'is_active': self.is_active
        }

# AES ENCRYPTION FOR SENSITIVE FIELDS

class EncryptedField(TypeDecorator):
    """SQLAlchemy type for AES-encrypted fields using Fernet."""
    impl = LargeBinary
    cache_ok = True
    
    def __init__(self, key=None):
        super().__init__()
        self._key = key
    
    @property
    def fernet(self):
        key = self._key or app.config['ENCRYPTION_KEY']
        if isinstance(key, str):
            key = key.encode()
        return Fernet(key)
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, dict):
            value = json.dumps(value)
        return self.fernet.encrypt(value.encode())
    
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        decrypted = self.fernet.decrypt(value).decode()
        try:
            return json.loads(decrypted)
        except json.JSONDecodeError:
            return decrypted

# DATABASE MODELS (TimescaleDB Hypertables)

class WastewaterSample(db.Model):
    """Wastewater surveillance data."""
    __tablename__ = 'wastewater_samples'
    
    id = db.Column(db.BigInteger, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    location_id = db.Column(db.String(50), nullable=False, index=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    pathogen_type = db.Column(db.String(100), nullable=False)
    viral_load_raw = db.Column(db.Float)  # copies/L
    viral_load_log = db.Column(db.Float)  # log10 transformed
    sample_volume_ml = db.Column(db.Float)
    ph_level = db.Column(db.Float)
    temperature_c = db.Column(db.Float)
    quality_score = db.Column(db.Float)  # 0-1 data quality indicator
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AirportSample(db.Model):
    """Airport wastewater and screening data."""
    __tablename__ = 'airport_samples'
    
    id = db.Column(db.BigInteger, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    airport_code = db.Column(db.String(10), nullable=False, index=True)
    airport_name = db.Column(db.String(200))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    flight_origin = db.Column(db.String(10))
    pathogen_type = db.Column(db.String(100), nullable=False)
    viral_load_raw = db.Column(db.Float)
    viral_load_log = db.Column(db.Float)
    passenger_count = db.Column(db.Integer)
    positive_screens = db.Column(db.Integer)
    sample_source = db.Column(db.String(50))  # 'aircraft_lavatory', 'terminal', 'screening'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ClinicalReport(db.Model):
    """Clinical diagnostic data with encrypted PII."""
    __tablename__ = 'clinical_reports'
    
    id = db.Column(db.BigInteger, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    facility_id = db.Column(db.String(50), nullable=False, index=True)
    region_code = db.Column(db.String(20), index=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    pathogen_type = db.Column(db.String(100), nullable=False)
    case_count = db.Column(db.Integer)
    hospitalized = db.Column(db.Integer)
    icu_admissions = db.Column(db.Integer)
    test_positivity_rate = db.Column(db.Float)

    # Encrypted sensitive fields (HIPAA compliance)

    patient_demographics = db.Column(EncryptedField())  # age ranges, etc.
    clinical_notes = db.Column(EncryptedField())  # anonymized but sensitive
    facility_details = db.Column(EncryptedField())  # internal facility data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnomalyAlert(db.Model):
    """ML-generated anomaly alerts."""
    __tablename__ = 'anomaly_alerts'
    
    id = db.Column(db.BigInteger, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    alert_type = db.Column(db.String(50), nullable=False)
    pathogen_type = db.Column(db.String(100))
    severity = db.Column(db.Float)  # 0-1 risk score
    location_id = db.Column(db.String(50))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    data_sources = db.Column(db.JSON)  # which streams triggered
    model_confidence = db.Column(db.Float)
    explanation = db.Column(db.Text)  # SHAP-based explanation
    acknowledged = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ETL PIPELINE

class ETLPipeline:
    """Pandas-based ETL for normalizing and geocoding pathogen data."""
    
    def __init__(self):
        self.geocoder = Nominatim(user_agent="pandemic_radar", timeout=10)
        self._geo_cache = {}
    
    def normalize_viral_load(self, df: pd.DataFrame, load_col: str = 'viral_load') -> pd.DataFrame:
        """Log-transform viral loads and add normalized columns."""
        df = df.copy()
        # Handle zeros and negatives for log transform
        df['viral_load_raw'] = df[load_col].clip(lower=1)
        df['viral_load_log'] = np.log10(df['viral_load_raw'])
        # Z-score normalization for ML input
        mean, std = df['viral_load_log'].mean(), df['viral_load_log'].std()
        df['viral_load_zscore'] = (df['viral_load_log'] - mean) / (std if std > 0 else 1)
        return df
    
    def geocode_location(self, query: str) -> tuple:
        """Geocode location string with caching."""
        if query in self._geo_cache:
            return self._geo_cache[query]
        try:
            loc = self.geocoder.geocode(query)
            result = (loc.latitude, loc.longitude) if loc else (None, None)
        except Exception:
            result = (None, None)
        self._geo_cache[query] = result
        return result
    
    def geocode_airports(self, df: pd.DataFrame, code_col: str = 'airport_code') -> pd.DataFrame:
        """Add lat/lon for airport codes."""
        df = df.copy()

        # Common airport coordinates

        airport_coords = {
            'JFK': (40.6413, -73.7781), 'LAX': (33.9425, -118.4081),
            'ORD': (41.9742, -87.9073), 'ATL': (33.6407, -84.4277),
            'LHR': (51.4700, -0.4543), 'CDG': (49.0097, 2.5479),
            'DXB': (25.2532, 55.3657), 'HND': (35.5494, 139.7798),
            'SIN': (1.3644, 103.9915), 'SYD': (-33.9399, 151.1753),
        }
        def get_coords(code):
            if code in airport_coords:
                return airport_coords[code]
            return self.geocode_location(f"{code} airport")
        
        coords = df[code_col].apply(get_coords)
        df['latitude'] = coords.apply(lambda x: x[0])
        df['longitude'] = coords.apply(lambda x: x[1])
        return df
    
    def process_wastewater(self, raw_data: list[dict]) -> pd.DataFrame:
        """Full ETL for wastewater samples."""
        df = pd.DataFrame(raw_data)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.utcnow()))
        df = self.normalize_viral_load(df, 'viral_load')
        # Quality score based on metadata completeness
        required = ['sample_volume_ml', 'ph_level', 'temperature_c']
        df['quality_score'] = df[required].notna().sum(axis=1) / len(required)
        return df
    
    def process_airport(self, raw_data: list[dict]) -> pd.DataFrame:
        """Full ETL for airport samples with geocoding."""
        df = pd.DataFrame(raw_data)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.utcnow()))
        df = self.normalize_viral_load(df, 'viral_load')
        df = self.geocode_airports(df, 'airport_code')
        return df
    
    def process_clinical(self, raw_data: list[dict]) -> pd.DataFrame:
        """ETL for clinical data (encryption handled at model level)."""
        df = pd.DataFrame(raw_data)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.utcnow()))
        # Calculate positivity rate if not provided
        if 'test_positivity_rate' not in df.columns and 'positive_tests' in df.columns:
            df['test_positivity_rate'] = df['positive_tests'] / df.get('total_tests', 1)
        return df

etl = ETLPipeline()

# CELERY ASYNC TASKS

@celery.task(bind=True, max_retries=3)
def ingest_wastewater_async(self, data: list[dict]):
    """Async wastewater data ingestion."""
    try:
        df = etl.process_wastewater(data)
        with app.app_context():
            for _, row in df.iterrows():
                sample = WastewaterSample(
                    timestamp=row['timestamp'],
                    location_id=row.get('location_id', 'unknown'),
                    latitude=row.get('latitude'),
                    longitude=row.get('longitude'),
                    pathogen_type=row.get('pathogen_type', 'unknown'),
                    viral_load_raw=row.get('viral_load_raw'),
                    viral_load_log=row.get('viral_load_log'),
                    sample_volume_ml=row.get('sample_volume_ml'),
                    ph_level=row.get('ph_level'),
                    temperature_c=row.get('temperature_c'),
                    quality_score=row.get('quality_score')
                )
                db.session.add(sample)
            db.session.commit()
        return {'status': 'success', 'records': len(df)}
    except Exception as e:
        self.retry(exc=e, countdown=60)

@celery.task(bind=True, max_retries=3)
def ingest_airport_async(self, data: list[dict]):
    """Async airport data ingestion."""
    try:
        df = etl.process_airport(data)
        with app.app_context():
            for _, row in df.iterrows():
                sample = AirportSample(
                    timestamp=row['timestamp'],
                    airport_code=row.get('airport_code', 'UNK'),
                    airport_name=row.get('airport_name'),
                    latitude=row.get('latitude'),
                    longitude=row.get('longitude'),
                    flight_origin=row.get('flight_origin'),
                    pathogen_type=row.get('pathogen_type', 'unknown'),
                    viral_load_raw=row.get('viral_load_raw'),
                    viral_load_log=row.get('viral_load_log'),
                    passenger_count=row.get('passenger_count'),
                    positive_screens=row.get('positive_screens'),
                    sample_source=row.get('sample_source')
                )
                db.session.add(sample)
            db.session.commit()
        return {'status': 'success', 'records': len(df)}
    except Exception as e:
        self.retry(exc=e, countdown=60)

@celery.task(bind=True, max_retries=3)
def ingest_clinical_async(self, data: list[dict]):
    """Async clinical data ingestion with encryption."""
    try:
        df = etl.process_clinical(data)
        with app.app_context():
            for _, row in df.iterrows():
                report = ClinicalReport(
                    timestamp=row['timestamp'],
                    facility_id=row.get('facility_id', 'unknown'),
                    region_code=row.get('region_code'),
                    latitude=row.get('latitude'),
                    longitude=row.get('longitude'),
                    pathogen_type=row.get('pathogen_type', 'unknown'),
                    case_count=row.get('case_count'),
                    hospitalized=row.get('hospitalized'),
                    icu_admissions=row.get('icu_admissions'),
                    test_positivity_rate=row.get('test_positivity_rate'),
                    # These fields auto-encrypt via EncryptedField
                    patient_demographics=row.get('patient_demographics'),
                    clinical_notes=row.get('clinical_notes'),
                    facility_details=row.get('facility_details')
                )
                db.session.add(report)
            db.session.commit()
        return {'status': 'success', 'records': len(df)}
    except Exception as e:
        self.retry(exc=e, countdown=60)

# API ROUTES

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

# AUTHENTICATION ROUTES

@app.route('/auth/register', methods=['POST'])
@limiter.limit("5/hour")
def register():
    """Register a new user (admin-only in production)."""
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    user = User(
        email=data['email'],
        role=data.get('role', 'viewer'),
        organization=data.get('organization')
    )
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created', 'user': user.to_dict()}), 201

@app.route('/auth/login', methods=['POST'])
@limiter.limit("10/minute")
def login():
    """Authenticate user and return JWT tokens."""
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account disabled'}), 403
    
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Create tokens with role claim
    additional_claims = {'role': user.role, 'org': user.organization}
    access_token = create_access_token(identity=user.id, additional_claims=additional_claims)
    refresh_token = create_refresh_token(identity=user.id, additional_claims=additional_claims)
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict(),
        'expires_in': app.config['JWT_ACCESS_TOKEN_EXPIRES'].total_seconds()
    })

@app.route('/auth/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token using refresh token."""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user or not user.is_active:
        return jsonify({'error': 'User not found or inactive'}), 401
    
    additional_claims = {'role': user.role, 'org': user.organization}
    access_token = create_access_token(identity=user_id, additional_claims=additional_claims)
    return jsonify({'access_token': access_token})

@app.route('/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    """Revoke current token."""
    jti = get_jwt()['jti']
    revoked_tokens.add(jti)
    return jsonify({'message': 'Token revoked'})

@app.route('/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current authenticated user info."""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'user': user.to_dict(), 'claims': get_jwt()})

# ADMIN ROUTES

@app.route('/admin/users', methods=['GET'])
@role_required('admin')
def list_users():
    """List all users (admin only)."""
    users = User.query.all()
    return jsonify({'users': [u.to_dict() for u in users]})

@app.route('/admin/users/<int:user_id>', methods=['PATCH'])
@role_required('admin')
def update_user(user_id):
    """Update user role or status (admin only)."""
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if 'role' in data:
        user.role = data['role']
    if 'is_active' in data:
        user.is_active = data['is_active']
    if 'organization' in data:
        user.organization = data['organization']
    
    db.session.commit()
    return jsonify({'user': user.to_dict()})

# --- Ingestion Endpoints ---

@app.route('/ingest/wastewater', methods=['POST'])
@limiter.limit("50/minute")
@role_required('admin', 'analyst', 'responder')
def ingest_wastewater():
    """Ingest wastewater surveillance data."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    samples = data if isinstance(data, list) else [data]
    task = ingest_wastewater_async.delay(samples)
    return jsonify({'status': 'accepted', 'task_id': task.id, 'records': len(samples)}), 202

@app.route('/ingest/airport', methods=['POST'])
@limiter.limit("50/minute")
@role_required('admin', 'analyst', 'responder')
def ingest_airport():
    """Ingest airport/aircraft surveillance data."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    samples = data if isinstance(data, list) else [data]
    task = ingest_airport_async.delay(samples)
    return jsonify({'status': 'accepted', 'task_id': task.id, 'records': len(samples)}), 202

@app.route('/ingest/clinical', methods=['POST'])
@limiter.limit("30/minute")
@role_required('admin', 'analyst')
def ingest_clinical():
    """Ingest clinical report data (encrypted storage). Restricted to admin/analyst."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    reports = data if isinstance(data, list) else [data]
    task = ingest_clinical_async.delay(reports)
    return jsonify({'status': 'accepted', 'task_id': task.id, 'records': len(reports)}), 202

# --- Query Endpoints ---

@app.route('/data/query', methods=['GET'])
@limiter.limit("100/minute")
def query_data():
    """Query aggregated pathogen data across sources."""
    pathogen = request.args.get('pathogen')
    start = request.args.get('start')  # ISO datetime
    end = request.args.get('end')
    source = request.args.get('source')  # wastewater, airport, clinical
    
    results = {'wastewater': [], 'airport': [], 'clinical': []}
    
    def build_query(model, base_query):
        q = base_query
        if pathogen:
            q = q.filter(model.pathogen_type.ilike(f'%{pathogen}%'))
        if start:
            q = q.filter(model.timestamp >= start)
        if end:
            q = q.filter(model.timestamp <= end)
        return q.order_by(model.timestamp.desc()).limit(1000)
    
    if not source or source == 'wastewater':
        q = build_query(WastewaterSample, WastewaterSample.query)
        results['wastewater'] = [{
            'timestamp': s.timestamp.isoformat(), 'location_id': s.location_id,
            'lat': s.latitude, 'lon': s.longitude, 'pathogen': s.pathogen_type,
            'viral_load_log': s.viral_load_log, 'quality': s.quality_score
        } for s in q.all()]
    
    if not source or source == 'airport':
        q = build_query(AirportSample, AirportSample.query)
        results['airport'] = [{
            'timestamp': s.timestamp.isoformat(), 'airport': s.airport_code,
            'lat': s.latitude, 'lon': s.longitude, 'pathogen': s.pathogen_type,
            'viral_load_log': s.viral_load_log, 'origin': s.flight_origin
        } for s in q.all()]
    
    if not source or source == 'clinical':
        q = build_query(ClinicalReport, ClinicalReport.query)
        results['clinical'] = [{
            'timestamp': r.timestamp.isoformat(), 'facility': r.facility_id,
            'region': r.region_code, 'lat': r.latitude, 'lon': r.longitude,
            'pathogen': r.pathogen_type, 'cases': r.case_count,
            'positivity': r.test_positivity_rate
            # Note: encrypted fields NOT exposed via API
        } for r in q.all()]
    
    return jsonify(results)

@app.route('/data/timeseries', methods=['GET'])
@limiter.limit("50/minute")
def get_timeseries():
    """Get time-series data optimized for ML and visualization."""
    pathogen = request.args.get('pathogen', 'SARS-CoV-2')
    days = int(request.args.get('days', 30))
    
    # TimescaleDB-optimized query using time_bucket
    query = text("""
        SELECT 
            time_bucket('1 day', timestamp) AS day,
            AVG(viral_load_log) AS avg_load,
            COUNT(*) AS sample_count,
            'wastewater' AS source
        FROM wastewater_samples
        WHERE pathogen_type ILIKE :pathogen
          AND timestamp > NOW() - INTERVAL ':days days'
        GROUP BY day
        UNION ALL
        SELECT 
            time_bucket('1 day', timestamp) AS day,
            AVG(viral_load_log) AS avg_load,
            COUNT(*) AS sample_count,
            'airport' AS source
        FROM airport_samples
        WHERE pathogen_type ILIKE :pathogen
          AND timestamp > NOW() - INTERVAL ':days days'
        GROUP BY day
        ORDER BY day DESC
    """)
    
    result = db.session.execute(query, {'pathogen': f'%{pathogen}%', 'days': days})
    return jsonify([dict(row._mapping) for row in result])

@app.route('/alerts', methods=['GET'])
@limiter.limit("100/minute")
def get_alerts():
    """Get recent anomaly alerts."""
    limit = int(request.args.get('limit', 50))
    unacked_only = request.args.get('unacked', 'false').lower() == 'true'
    
    q = AnomalyAlert.query.order_by(AnomalyAlert.timestamp.desc())
    if unacked_only:
        q = q.filter(AnomalyAlert.acknowledged == False)
    
    alerts = q.limit(limit).all()
    return jsonify([{
        'id': a.id, 'timestamp': a.timestamp.isoformat(),
        'type': a.alert_type, 'pathogen': a.pathogen_type,
        'severity': a.severity, 'location': a.location_id,
        'lat': a.latitude, 'lon': a.longitude,
        'sources': a.data_sources, 'confidence': a.model_confidence,
        'explanation': a.explanation, 'acknowledged': a.acknowledged
    } for a in alerts])

@app.route('/alerts/<int:alert_id>/ack', methods=['POST'])
@limiter.limit("50/minute")
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    alert = AnomalyAlert.query.get_or_404(alert_id)
    alert.acknowledged = True
    db.session.commit()
    return jsonify({'status': 'acknowledged', 'id': alert_id})

# --- ML Integration Endpoint ---

@app.route('/ml/predict', methods=['POST'])
@limiter.limit("20/minute")
def ml_predict():
    """Endpoint for ML model to submit predictions/alerts."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    alert = AnomalyAlert(
        timestamp=datetime.utcnow(),
        alert_type=data.get('alert_type', 'anomaly'),
        pathogen_type=data.get('pathogen_type'),
        severity=data.get('severity', 0.5),
        location_id=data.get('location_id'),
        latitude=data.get('latitude'),
        longitude=data.get('longitude'),
        data_sources=data.get('data_sources', []),
        model_confidence=data.get('confidence'),
        explanation=data.get('explanation')
    )
    db.session.add(alert)
    db.session.commit()
    
    return jsonify({'status': 'created', 'alert_id': alert.id}), 201

# DATABASE INITIALIZATION (TimescaleDB Hypertables)

def init_db():
    """Initialize database with TimescaleDB hypertables."""
    with app.app_context():
        db.create_all()
        # Convert to TimescaleDB hypertables for time-series optimization
        hypertables = [
            ('wastewater_samples', 'timestamp'),
            ('airport_samples', 'timestamp'),
            ('clinical_reports', 'timestamp'),
            ('anomaly_alerts', 'timestamp')
        ]
        for table, time_col in hypertables:
            try:
                db.session.execute(text(
                    f"SELECT create_hypertable('{table}', '{time_col}', if_not_exists => TRUE)"
                ))
            except Exception as e:
                print(f"Hypertable {table}: {e}")  # May fail if not TimescaleDB
        db.session.commit()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
