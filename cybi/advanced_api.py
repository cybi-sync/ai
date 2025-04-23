from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query, Path, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from pydantic import BaseModel, Field, validator, root_validator
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import logging
import os
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime, timedelta
import io
from uuid import uuid4
import asyncio
import time
import jwt
from jose import JWTError, jwt as jose_jwt
from passlib.context import CryptContext
import aiofiles
from starlette.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from contextlib import asynccontextmanager
import socketio
import csv
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing
import aiohttp

from cybi.data_module import CYBIDataProcessor
from cybi.main import CYBISystem
from cybi.model_architecture import CYBIModelBuilder
from cybi.training_module import CYBIModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cybi_advanced_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Secret key for JWT token
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI app
app = FastAPI(
    title="CYBI Smartshoe AI Advanced API",
    description="Advanced API for weight prediction, health monitoring, and system management from CYBI smartshoe data",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Socket.IO
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio)
app.mount("/ws", socket_app)

# Initialize thread pool
thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

# Initialize CYBI system
cybi_system = None

# Model cache
models = {
    'weight_model': None,
    'health_model': None,
    'is_loaded': False,
    'last_updated': None
}

# Job tracking
active_jobs = {}

# Database setup
DATABASE_URL = "sqlite:///./cybi_api.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    api_key = Column(String, unique=True, index=True)
    
    profiles = relationship("UserProfile", back_populates="user")
    devices = relationship("Device", back_populates="user")
    predictions = relationship("PredictionHistory", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    age = Column(Integer, nullable=True)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    gender = Column(String, nullable=True)
    medical_conditions = Column(JSON, nullable=True)
    activity_level = Column(String, nullable=True)
    shoe_size = Column(String, nullable=True)
    
    user = relationship("User", back_populates="profiles")

class Device(Base):
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_type = Column(String)
    firmware_version = Column(String)
    last_connected = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    config = Column(JSON, nullable=True)
    
    user = relationship("User", back_populates="devices")
    data_sessions = relationship("DataSession", back_populates="device")

class DataSession(Base):
    __tablename__ = "data_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="active")
    data_points = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)
    
    device = relationship("Device", back_populates="data_sessions")
    sensor_data = relationship("SensorDataPoint", back_populates="session")

class SensorDataPoint(Base):
    __tablename__ = "sensor_data_points"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("data_sessions.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    pressure_data = Column(JSON)
    accelerometer_data = Column(JSON)
    gyroscope_data = Column(JSON)
    additional_data = Column(JSON, nullable=True)
    
    session = relationship("DataSession", back_populates="sensor_data")

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    weight_prediction = Column(Float, nullable=True)
    health_prediction = Column(JSON, nullable=True)
    confidence_scores = Column(JSON, nullable=True)
    feedback = Column(JSON, nullable=True)
    
    user = relationship("User", back_populates="predictions")

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    job_type = Column(String)  # weight, health, both
    status = Column(String, default="queued")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    configuration = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)  # weight, health
    version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    trained_by = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    file_path = Column(String)
    
class SystemConfig(Base):
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(JSON)
    description = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create database tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models (Schemas)
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    is_admin: bool = False

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None

class UserInDB(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    api_key: str
    
    class Config:
        orm_mode = True

class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class ProfileBase(BaseModel):
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    gender: Optional[str] = None
    medical_conditions: Optional[List[str]] = None
    activity_level: Optional[str] = None
    shoe_size: Optional[str] = None

class ProfileCreate(ProfileBase):
    pass

class ProfileUpdate(ProfileBase):
    pass

class ProfileResponse(ProfileBase):
    id: int
    user_id: int
    
    class Config:
        orm_mode = True

class DeviceBase(BaseModel):
    device_id: str
    device_type: str
    firmware_version: str
    config: Optional[Dict[str, Any]] = None

class DeviceCreate(DeviceBase):
    pass

class DeviceUpdate(BaseModel):
    firmware_version: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

class DeviceResponse(DeviceBase):
    id: int
    user_id: int
    last_connected: Optional[datetime] = None
    is_active: bool
    
    class Config:
        orm_mode = True

class SensorDataBase(BaseModel):
    """Model for incoming sensor data from the smartshoe."""
    timestamp: float
    pressure_sensors: List[float] = Field(..., description="Readings from pressure sensors")
    accelerometer: Dict[str, List[float]] = Field(..., description="Accelerometer readings (x, y, z) for left and right shoe")
    gyroscope: Dict[str, List[float]] = Field(..., description="Gyroscope readings (x, y, z) for left and right shoe")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Any additional sensor data")

class DataSessionCreate(BaseModel):
    device_id: str
    metadata: Optional[Dict[str, Any]] = None

class DataSessionResponse(BaseModel):
    session_id: str
    device_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    data_points: int
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True

class DataSessionUpdate(BaseModel):
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    """Model for batch prediction requests."""
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    data: List[SensorDataBase]

class PredictionResult(BaseModel):
    """Model for prediction results."""
    timestamp: float
    weight_prediction: float
    health_prediction: Dict[str, Any]
    confidence: Dict[str, float]

class PredictionResponse(BaseModel):
    """Model for the overall prediction response."""
    request_id: str
    results: List[PredictionResult]
    processing_time: float
    status: str

class FeedbackRequest(BaseModel):
    """Model for providing feedback on predictions."""
    prediction_id: int
    actual_weight: Optional[float] = None
    actual_health_condition: Optional[str] = None
    feedback_notes: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")

class TrainingRequest(BaseModel):
    """Model for requesting model training."""
    model_type: str = Field(..., description="Type of model to train: 'weight', 'health', or 'both'")
    configuration: Optional[Dict[str, Any]] = None
    generate_synthetic_data: bool = False
    data_session_ids: Optional[List[str]] = None
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['weight', 'health', 'both']:
            raise ValueError('model_type must be one of: weight, health, both')
        return v

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class TrainingJobDetail(TrainingJobResponse):
    configuration: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    class Config:
        orm_mode = True

class ModelVersionResponse(BaseModel):
    id: int
    model_type: str
    version: str
    created_at: datetime
    trained_by: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    is_active: bool
    
    class Config:
        orm_mode = True

class StatusResponse(BaseModel):
    """Model for API status responses."""
    status: str
    version: str
    models_loaded: bool
    active_model_versions: Dict[str, str]
    available_model_versions: Dict[str, List[str]]
    device_count: int
    uptime: float
    timestamp: float

class SystemConfigUpdate(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None

class SystemConfigResponse(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None
    updated_at: datetime
    
    class Config:
        orm_mode = True

class AnalyticsRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_id: Optional[int] = None
    device_id: Optional[str] = None
    metrics: List[str] = Field(default=["weight", "health", "activity"])
    
    @validator('metrics')
    def validate_metrics(cls, v):
        allowed_metrics = ["weight", "health", "activity", "gait", "pressure"]
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Invalid metric: {metric}. Allowed values: {allowed_metrics}")
        return v

class AnalyticsResponse(BaseModel):
    request_id: str
    metrics: Dict[str, Any]
    period: Dict[str, datetime]

# Authentication and Security Functions
def verify_password(plain_password, hashed_password):
    """Verify password against hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token for user authentication."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jose_jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def get_user_by_username(db: Session, username: str):
    """Get user by username from the database."""
    return db.query(User).filter(User.username == username).first()

def get_user_by_api_key(db: Session, api_key: str):
    """Get user by API key from the database."""
    return db.query(User).filter(User.api_key == api_key).first()

def authenticate_user(db: Session, username: str, password: str):
    """Authenticate user with username and password."""
    user = get_user_by_username(db, username)
    
    if not user:
        return False
    
    if not verify_password(password, user.hashed_password):
        return False
    
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username, is_admin=payload.get("is_admin", False))
    
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_username(db, username=token_data.username)
    
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if current user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user

async def get_admin_user(current_user: User = Depends(get_current_active_user)):
    """Check if current user is an admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    return current_user

async def validate_api_key(api_key: str = Security(API_KEY_HEADER), db: Session = Depends(get_db)):
    """Validate API key and get the user."""
    user = get_user_by_api_key(db, api_key)
    
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    
    return user

# Authentication and User Management Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Get an access token for API authentication using username and password."""
    user = authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "is_admin": user.is_admin},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    """Create a new user (admin only)."""
    # Check if user already exists
    db_user = get_user_by_username(db, username=user.username)
    
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create new user
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        is_active=True,
        is_admin=False,
        api_key=str(uuid4())
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@app.get("/users", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    """Get list of users (admin only)."""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get("/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@app.put("/users/me", response_model=UserResponse)
async def update_user_info(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update current user information."""
    # Update email if provided
    if user_update.email is not None:
        current_user.email = user_update.email
    
    # Update password if provided
    if user_update.password is not None:
        current_user.hashed_password = get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(current_user)
    
    return current_user

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user information by ID (admin or self only)."""
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.put("/users/{user_id}/activate", response_model=UserResponse)
async def activate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """Activate or deactivate a user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Toggle active status
    user.is_active = not user.is_active
    db.commit()
    db.refresh(user)
    
    return user

@app.put("/users/{user_id}/admin", response_model=UserResponse)
async def toggle_admin(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """Toggle admin status for a user (admin only)."""
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot modify your own admin status")
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Toggle admin status
    user.is_admin = not user.is_admin
    db.commit()
    db.refresh(user)
    
    return user

@app.post("/users/{user_id}/api-key", response_model=dict)
async def regenerate_api_key(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Regenerate API key for a user (admin or self only)."""
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate new API key
    user.api_key = str(uuid4())
    db.commit()
    db.refresh(user)
    
    return {"api_key": user.api_key}

# User Profile Endpoints
@app.post("/users/{user_id}/profile", response_model=ProfileResponse)
async def create_user_profile(
    user_id: int,
    profile: ProfileCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a profile for a user (admin or self only)."""
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Check if user exists
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if profile already exists
    existing_profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    if existing_profile:
        raise HTTPException(status_code=400, detail="Profile already exists")
    
    # Create profile
    new_profile = UserProfile(
        user_id=user_id,
        age=profile.age,
        height=profile.height,
        weight=profile.weight,
        gender=profile.gender,
        medical_conditions=profile.medical_conditions,
        activity_level=profile.activity_level,
        shoe_size=profile.shoe_size
    )
    
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    
    return new_profile

@app.get("/users/{user_id}/profile", response_model=ProfileResponse)
async def get_user_profile(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a user's profile (admin or self only)."""
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

@app.put("/users/{user_id}/profile", response_model=ProfileResponse)
async def update_user_profile(
    user_id: int,
    profile_update: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a user's profile (admin or self only)."""
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Update profile fields
    for key, value in profile_update.dict(exclude_unset=True).items():
        setattr(profile, key, value)
    
    db.commit()
    db.refresh(profile)
    
    return profile

# Device Management Endpoints
@app.post("/devices", response_model=DeviceResponse)
async def register_device(
    device: DeviceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Register a new device for the current user."""
    # Check if device already exists
    existing_device = db.query(Device).filter(Device.device_id == device.device_id).first()
    
    if existing_device:
        raise HTTPException(status_code=400, detail="Device already registered")
    
    # Create new device
    new_device = Device(
        device_id=device.device_id,
        user_id=current_user.id,
        device_type=device.device_type,
        firmware_version=device.firmware_version,
        last_connected=datetime.utcnow(),
        is_active=True,
        config=device.config
    )
    
    db.add(new_device)
    db.commit()
    db.refresh(new_device)
    
    return new_device

@app.get("/devices", response_model=List[DeviceResponse])
async def get_devices(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of devices for the current user (or all devices for admin)."""
    if current_user.is_admin:
        devices = db.query(Device).offset(skip).limit(limit).all()
    else:
        devices = db.query(Device).filter(Device.user_id == current_user.id).offset(skip).limit(limit).all()
    
    return devices

@app.get("/devices/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get device information by ID (admin or device owner only)."""
    device = db.query(Device).filter(Device.device_id == device_id).first()
    
    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")
    
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    return device

@app.put("/devices/{device_id}", response_model=DeviceResponse)
async def update_device(
    device_id: str,
    device_update: DeviceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update device information (admin or device owner only)."""
    device = db.query(Device).filter(Device.device_id == device_id).first()
    
    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")
    
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Update device fields
    update_data = device_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(device, key, value)
    
    # Update last connected timestamp
    device.last_connected = datetime.utcnow()
    
    db.commit()
    db.refresh(device)
    
    return device

# Data Collection Endpoints
@app.post("/data-sessions", response_model=DataSessionResponse)
async def create_data_session(
    session_data: DataSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new data collection session."""
    # Check if device exists and belongs to the user
    device = db.query(Device).filter(Device.device_id == session_data.device_id).first()
    
    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")
    
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Create new session
    session_id = str(uuid4())
    new_session = DataSession(
        session_id=session_id,
        device_id=device.id,
        start_time=datetime.utcnow(),
        status="active",
        metadata=session_data.metadata
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return new_session

@app.get("/data-sessions", response_model=List[DataSessionResponse])
async def get_data_sessions(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    device_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of data collection sessions for the current user (or all sessions for admin)."""
    query = db.query(DataSession)
    
    # Apply filters
    if status:
        query = query.filter(DataSession.status == status)
    
    if device_id:
        device = db.query(Device).filter(Device.device_id == device_id).first()
        if device:
            query = query.filter(DataSession.device_id == device.id)
    
    # Filter by user
    if not current_user.is_admin:
        query = query.join(Device).filter(Device.user_id == current_user.id)
    
    sessions = query.offset(skip).limit(limit).all()
    return sessions

@app.get("/data-sessions/{session_id}", response_model=DataSessionResponse)
async def get_data_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get data session information by ID (admin or device owner only)."""
    session = db.query(DataSession).filter(DataSession.session_id == session_id).first()
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check permissions
    device = db.query(Device).filter(Device.id == session.device_id).first()
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    return session

@app.put("/data-sessions/{session_id}", response_model=DataSessionResponse)
async def update_data_session(
    session_id: str,
    session_update: DataSessionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update data session information (admin or device owner only)."""
    session = db.query(DataSession).filter(DataSession.session_id == session_id).first()
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check permissions
    device = db.query(Device).filter(Device.id == session.device_id).first()
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Update session fields
    update_data = session_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(session, key, value)
    
    # If status is being set to 'completed', set the end time
    if session_update.status == "completed" and session.end_time is None:
        session.end_time = datetime.utcnow()
    
    db.commit()
    db.refresh(session)
    
    return session

@app.post("/data-sessions/{session_id}/data")
async def add_sensor_data(
    session_id: str,
    sensor_data: List[SensorDataBase],
    db: Session = Depends(get_db),
    current_user: User = Security(validate_api_key)
):
    """Add sensor data to a session."""
    session = db.query(DataSession).filter(DataSession.session_id == session_id).first()
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Check permissions
    device = db.query(Device).filter(Device.id == session.device_id).first()
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Add data points
    for data_point in sensor_data:
        new_data_point = SensorDataPoint(
            session_id=session.id,
            timestamp=datetime.fromtimestamp(data_point.timestamp),
            pressure_data=data_point.pressure_sensors,
            accelerometer_data=data_point.accelerometer,
            gyroscope_data=data_point.gyroscope,
            additional_data=data_point.additional_data
        )
        db.add(new_data_point)
    
    # Update data point count
    session.data_points += len(sensor_data)
    
    db.commit()
    
    # Send real-time update via WebSocket
    await sio.emit(
        'data_update',
        {
            'session_id': session_id,
            'device_id': device.device_id,
            'data_points': session.data_points,
            'timestamp': datetime.utcnow().isoformat()
        },
        room=f"device_{device.device_id}"
    )
    
    return {"status": "success", "data_points_added": len(sensor_data)}

@app.get("/data-sessions/{session_id}/data")
async def get_session_data(
    session_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get sensor data for a session (admin or device owner only)."""
    session = db.query(DataSession).filter(DataSession.session_id == session_id).first()
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check permissions
    device = db.query(Device).filter(Device.id == session.device_id).first()
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Get data points
    data_points = db.query(SensorDataPoint).filter(
        SensorDataPoint.session_id == session.id
    ).order_by(
        SensorDataPoint.timestamp
    ).offset(skip).limit(limit).all()
    
    # Format data
    formatted_data = []
    for point in data_points:
        formatted_data.append({
            "timestamp": point.timestamp.timestamp(),
            "pressure_sensors": point.pressure_data,
            "accelerometer": point.accelerometer_data,
            "gyroscope": point.gyroscope_data,
            "additional_data": point.additional_data
        })
    
    return {"session_id": session_id, "data": formatted_data, "total_points": session.data_points}

@app.get("/data-sessions/{session_id}/export")
async def export_session_data(
    session_id: str,
    format: str = Query("csv", regex="^(csv|json)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Export session data in CSV or JSON format (admin or device owner only)."""
    session = db.query(DataSession).filter(DataSession.session_id == session_id).first()
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check permissions
    device = db.query(Device).filter(Device.id == session.device_id).first()
    if not current_user.is_admin and device.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    
    # Get all data points
    data_points = db.query(SensorDataPoint).filter(
        SensorDataPoint.session_id == session.id
    ).order_by(
        SensorDataPoint.timestamp
    ).all()
    
    if format == "csv":
        # Create CSV file in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        header = ["timestamp", "sensor_type", "sensor_location", "dimension", "value"]
        writer.writerow(header)
        
        # Write data
        for point in data_points:
            timestamp = point.timestamp.timestamp()
            
            # Write pressure data
            for i, value in enumerate(point.pressure_data):
                sensor_location = "left" if i < 6 else "right"
                position = ["heel", "arch", "mid", "ball", "toe", "edge"][i % 6]
                writer.writerow([timestamp, "pressure", f"{sensor_location}_{position}", "magnitude", value])
            
            # Write accelerometer data
            for side, values in point.accelerometer_data.items():
                for i, axis in enumerate(["x", "y", "z"]):
                    if i < len(values):
                        writer.writerow([timestamp, "accelerometer", f"{side}_{axis}", "magnitude", values[i]])
            
            # Write gyroscope data
            for side, values in point.gyroscope_data.items():
                for i, axis in enumerate(["x", "y", "z"]):
                    if i < len(values):
                        writer.writerow([timestamp, "gyroscope", f"{side}_{axis}", "magnitude", values[i]])
            
            # Write additional data
            for key, value in point.additional_data.items():
                writer.writerow([timestamp, "additional", key, "value", value])
    
    elif format == "json":
        # Convert data points to JSON
        formatted_data = []
        for point in data_points:
            formatted_data.append({
                "timestamp": point.timestamp.timestamp(),
                "sensor_type": "pressure",
                "sensor_location": "left" if i < 6 else "right",
                "dimension": ["heel", "arch", "mid", "ball", "toe", "edge"][i % 6],
                "value": point.pressure_data[i]
            })
            for side, values in point.accelerometer_data.items():
                for i, axis in enumerate(["x", "y", "z"]):
                    if i < len(values):
                        formatted_data.append({
                            "timestamp": point.timestamp.timestamp(),
                            "sensor_type": "accelerometer",
                            "sensor_location": side,
                            "dimension": axis,
                            "value": values[i]
                        })
            for side, values in point.gyroscope_data.items():
                for i, axis in enumerate(["x", "y", "z"]):
                    if i < len(values):
                        formatted_data.append({
                            "timestamp": point.timestamp.timestamp(),
                            "sensor_type": "gyroscope",
                            "sensor_location": side,
                            "dimension": axis,
                            "value": values[i]
                        })
            for key, value in point.additional_data.items():
                formatted_data.append({
                    "timestamp": point.timestamp.timestamp(),
                    "sensor_type": "additional",
                    "sensor_location": key,
                    "dimension": "value",
                    "value": value
                })
    
    return JSONResponse(content={"session_id": session_id, "data": formatted_data})

# System Management Endpoints
@app.get("/status", response_model=StatusResponse)
async def get_status(db: Session = Depends(get_db)):
    """Get system status information."""
    # Count active devices
    device_count = db.query(Device).filter(Device.is_active == True).count()
    
    # Get active model versions
    weight_model = db.query(ModelVersion).filter(
        ModelVersion.model_type == "weight",
        ModelVersion.is_active == True
    ).order_by(ModelVersion.created_at.desc()).first()
    
    health_model = db.query(ModelVersion).filter(
        ModelVersion.model_type == "health",
        ModelVersion.is_active == True
    ).order_by(ModelVersion.created_at.desc()).first()
    
    active_versions = {
        "weight": weight_model.version if weight_model else "none",
        "health": health_model.version if health_model else "none"
    }
    
    # Get all available model versions
    available_versions = {}
    
    weight_versions = db.query(ModelVersion.version).filter(
        ModelVersion.model_type == "weight"
    ).order_by(ModelVersion.created_at.desc()).limit(10).all()
    
    health_versions = db.query(ModelVersion.version).filter(
        ModelVersion.model_type == "health"
    ).order_by(ModelVersion.created_at.desc()).limit(10).all()
    
    available_versions["weight"] = [v[0] for v in weight_versions]
    available_versions["health"] = [v[0] for v in health_versions]
    
    return {
        "status": "operational" if models['is_loaded'] else "degraded",
        "version": "2.0.0",
        "models_loaded": models['is_loaded'],
        "active_model_versions": active_versions,
        "available_model_versions": available_versions,
        "device_count": device_count,
        "uptime": time.time() - start_time,
        "timestamp": time.time()
    }

@app.get("/system/config", response_model=List[SystemConfigResponse])
async def get_system_config(db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    """Get system configuration (admin only)."""
    configs = db.query(SystemConfig).all()
    return configs

@app.post("/system/config", response_model=SystemConfigResponse)
async def update_system_config(
    config: SystemConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """Update system configuration (admin only)."""
    existing_config = db.query(SystemConfig).filter(SystemConfig.key == config.key).first()
    
    if existing_config:
        # Update existing config
        existing_config.value = config.value
        
        if config.description:
            existing_config.description = config.description
        
        existing_config.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_config)
        
        return existing_config
    else:
        # Create new config
        new_config = SystemConfig(
            key=config.key,
            value=config.value,
            description=config.description
        )
        
        db.add(new_config)
        db.commit()
        db.refresh(new_config)
        
        return new_config

@app.post("/system/reload-models")
async def reload_models(current_user: User = Depends(get_admin_user)):
    """Reload models from disk (admin only)."""
    global models
    
    await load_models_async()
    
    return {"status": "success", "message": "Models reloading initiated"}

# Initialize global state
start_time = time.time()

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advanced_api:app", host="0.0.0.0", port=8000, reload=True)