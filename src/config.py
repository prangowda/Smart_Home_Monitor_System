"""
Configuration settings for the Smart Home Monitoring System
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# General Settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Sensor Settings
SENSOR_TYPES = [
    'temperature',
    'humidity',
    'motion',
    'light',
    'door',
    'window',
    'smoke',
    'co2'
]

SENSOR_LOCATIONS = [
    'living_room',
    'kitchen',
    'bedroom',
    'bathroom',
    'entrance',
    'garage'
]

# Sensor data generation settings
SENSOR_UPDATE_INTERVAL = 2  # seconds
ANOMALY_PROBABILITY = 0.05  # 5% chance of generating anomalous data

# Machine Learning Settings
ANOMALY_THRESHOLD = 0.8
MODEL_UPDATE_INTERVAL = 3600  # seconds (1 hour)
TRAINING_DATA_WINDOW = 86400  # seconds (24 hours)

# API Settings
API_HOST = os.getenv('API_HOST', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', 5000))

# Dashboard Settings
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '127.0.0.1')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8050))
DASHBOARD_UPDATE_INTERVAL = 1  # seconds

# Create data and model directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
