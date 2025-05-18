"""
REST API for Smart Home Monitoring System

This module provides a Flask-based REST API for accessing sensor data
and anomaly detection results.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_HOST, API_PORT
from sensors.sensor_simulator import SensorManager
from ml.anomaly_detector import AnomalyDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-home-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize sensor manager and anomaly detector
sensor_manager = SensorManager()
anomaly_detector = AnomalyDetector()

# In-memory cache for recent sensor readings and anomalies
recent_readings = {}
recent_anomalies = []
MAX_RECENT_READINGS = 1000  # Per sensor type
MAX_RECENT_ANOMALIES = 100

@app.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        'name': 'Smart Home Monitoring System API',
        'version': '1.0',
        'endpoints': [
            '/api/sensors',
            '/api/sensors/<sensor_type>',
            '/api/sensors/location/<location>',
            '/api/sensors/<sensor_id>',
            '/api/anomalies',
            '/api/anomalies/<sensor_type>',
            '/api/stats'
        ]
    })

@app.route('/api/sensors')
def get_all_sensors():
    """Get all available sensors."""
    sensors = sensor_manager.get_all_sensors()
    return jsonify([{
        'sensor_id': sensor.sensor_id,
        'sensor_type': sensor.sensor_type,
        'location': sensor.location
    } for sensor in sensors])

@app.route('/api/sensors/<sensor_type>')
def get_sensors_by_type(sensor_type):
    """Get sensors of a specific type."""
    sensors = sensor_manager.get_sensors_by_type(sensor_type)
    return jsonify([{
        'sensor_id': sensor.sensor_id,
        'sensor_type': sensor.sensor_type,
        'location': sensor.location
    } for sensor in sensors])

@app.route('/api/sensors/location/<location>')
def get_sensors_by_location(location):
    """Get sensors in a specific location."""
    sensors = sensor_manager.get_sensors_by_location(location)
    return jsonify([{
        'sensor_id': sensor.sensor_id,
        'sensor_type': sensor.sensor_type,
        'location': sensor.location
    } for sensor in sensors])

@app.route('/api/sensors/<sensor_id>')
def get_sensor_readings(sensor_id):
    """
    Get readings for a specific sensor.
    
    Query parameters:
    - limit: Maximum number of readings to return (default: 100)
    - from_time: Start time as ISO format or seconds since epoch
    - to_time: End time as ISO format or seconds since epoch
    """
    # Parse query parameters
    limit = min(int(request.args.get('limit', 100)), 1000)
    
    # Get cached readings for this sensor
    sensor_readings = recent_readings.get(sensor_id, [])
    
    # Apply time filters if provided
    if 'from_time' in request.args:
        from_time = parse_time(request.args.get('from_time'))
        sensor_readings = [r for r in sensor_readings if r['timestamp'] >= from_time]
    
    if 'to_time' in request.args:
        to_time = parse_time(request.args.get('to_time'))
        sensor_readings = [r for r in sensor_readings if r['timestamp'] <= to_time]
    
    # Limit the number of results
    sensor_readings = sensor_readings[-limit:]
    
    return jsonify(sensor_readings)

@app.route('/api/anomalies')
def get_anomalies():
    """
    Get detected anomalies.
    
    Query parameters:
    - limit: Maximum number of anomalies to return (default: 100)
    - from_time: Start time as ISO format or seconds since epoch
    - to_time: End time as ISO format or seconds since epoch
    - sensor_type: Filter by sensor type
    - location: Filter by location
    """
    # Parse query parameters
    limit = min(int(request.args.get('limit', 100)), 1000)
    sensor_type = request.args.get('sensor_type')
    location = request.args.get('location')
    
    # Get cached anomalies
    anomalies = recent_anomalies.copy()
    
    # Apply filters
    if sensor_type:
        anomalies = [a for a in anomalies if a['sensor_type'] == sensor_type]
    
    if location:
        anomalies = [a for a in anomalies if a['location'] == location]
    
    # Apply time filters if provided
    if 'from_time' in request.args:
        from_time = parse_time(request.args.get('from_time'))
        anomalies = [a for a in anomalies if a['timestamp'] >= from_time]
    
    if 'to_time' in request.args:
        to_time = parse_time(request.args.get('to_time'))
        anomalies = [a for a in anomalies if a['timestamp'] <= to_time]
    
    # Limit the number of results
    anomalies = anomalies[-limit:]
    
    return jsonify(anomalies)

@app.route('/api/anomalies/<sensor_type>')
def get_anomalies_by_type(sensor_type):
    """Get anomalies for a specific sensor type."""
    request.args = dict(request.args)
    request.args['sensor_type'] = sensor_type
    return get_anomalies()

@app.route('/api/stats')
def get_stats():
    """Get system statistics."""
    # Count sensors by type
    sensor_counts = {}
    for sensor_type in set(s.sensor_type for s in sensor_manager.get_all_sensors()):
        sensor_counts[sensor_type] = len(sensor_manager.get_sensors_by_type(sensor_type))
    
    # Count recent readings and anomalies
    reading_counts = {k: len(v) for k, v in recent_readings.items()}
    
    return jsonify({
        'sensors': {
            'total': len(sensor_manager.get_all_sensors()),
            'by_type': sensor_counts
        },
        'readings': {
            'total': sum(len(v) for v in recent_readings.values()),
            'by_sensor': reading_counts
        },
        'anomalies': {
            'total': len(recent_anomalies),
            'recent_count': sum(1 for a in recent_anomalies if a['timestamp'] > time.time() - 3600)
        },
        'system': {
            'uptime': time.time() - start_time
        }
    })

@app.route('/api/train', methods=['POST'])
def train_models():
    """Trigger model training."""
    force = request.json.get('force', False) if request.is_json else False
    
    success = anomaly_detector.train_models(force=force)
    
    return jsonify({
        'success': success,
        'message': 'Training started' if success else 'Training already in progress'
    })

def parse_time(time_str):
    """
    Parse a time string to a Unix timestamp.
    
    Args:
        time_str (str): Time string as ISO format or seconds since epoch
        
    Returns:
        float: Unix timestamp
    """
    try:
        # Try parsing as a float (seconds since epoch)
        return float(time_str)
    except ValueError:
        try:
            # Try parsing as ISO format
            dt = datetime.fromisoformat(time_str)
            return dt.timestamp()
        except ValueError:
            # Default to 24 hours ago
            return time.time() - 86400

def process_sensor_readings(readings):
    """
    Process new sensor readings: cache them and detect anomalies.
    
    Args:
        readings (list): List of sensor reading objects or dictionaries
    """
    # Convert readings to dictionaries if they are objects
    reading_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in readings]
    
    # Cache readings by sensor ID
    for reading in reading_dicts:
        sensor_id = reading['sensor_id']
        if sensor_id not in recent_readings:
            recent_readings[sensor_id] = []
        
        recent_readings[sensor_id].append(reading)
        
        # Limit the number of cached readings
        if len(recent_readings[sensor_id]) > MAX_RECENT_READINGS:
            recent_readings[sensor_id] = recent_readings[sensor_id][-MAX_RECENT_READINGS:]
    
    # Detect anomalies
    detected_anomalies = anomaly_detector.detect_anomalies(reading_dicts)
    
    # Cache anomalies
    for result in detected_anomalies:
        if result.get('is_anomaly', False):
            # Add to recent anomalies
            recent_anomalies.append(result)
            
            # Limit the number of cached anomalies
            if len(recent_anomalies) > MAX_RECENT_ANOMALIES:
                recent_anomalies.pop(0)
            
            # Emit via WebSocket
            socketio.emit('anomaly', result)
    
    # Emit new readings via WebSocket
    socketio.emit('readings', reading_dicts)

def start_api_server(host=None, port=None):
    """
    Start the API server.
    
    Args:
        host (str, optional): Host to bind to
        port (int, optional): Port to bind to
    """
    global start_time
    start_time = time.time()
    
    host = host or API_HOST
    port = port or API_PORT
    
    # Start the sensor simulation
    sensor_manager.start_simulation(callback=process_sensor_readings)
    
    # Start the API server
    logger.info(f"Starting API server on {host}:{port}")
    socketio.run(app, host=host, port=port)

def stop_api_server():
    """Stop the API server and sensor simulation."""
    sensor_manager.stop_simulation()
    logger.info("API server stopped")


# For testing
if __name__ == '__main__':
    start_api_server()
