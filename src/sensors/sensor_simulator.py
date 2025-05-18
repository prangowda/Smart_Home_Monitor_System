"""
Sensor Simulator for Smart Home Monitoring System

This module simulates various IoT sensors in a smart home environment,
generating realistic data with occasional anomalies for testing the
anomaly detection system.
"""
import time
import random
import json
import threading
import logging
from datetime import datetime
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SENSOR_TYPES, SENSOR_LOCATIONS, SENSOR_UPDATE_INTERVAL,
    ANOMALY_PROBABILITY, DATA_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sensor_simulator')

class SensorReading:
    """Represents a single sensor reading."""
    
    def __init__(self, sensor_id, sensor_type, location, value, timestamp=None, is_anomaly=False):
        """
        Initialize a sensor reading.
        
        Args:
            sensor_id (str): Unique identifier for the sensor
            sensor_type (str): Type of sensor (e.g., temperature, humidity)
            location (str): Location of the sensor in the home
            value (float): The sensor reading value
            timestamp (float, optional): Unix timestamp of the reading
            is_anomaly (bool, optional): Whether this reading is anomalous
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.location = location
        self.value = value
        self.timestamp = timestamp or time.time()
        self.is_anomaly = is_anomaly
    
    def to_dict(self):
        """Convert the sensor reading to a dictionary."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'location': self.location,
            'value': self.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'is_anomaly': self.is_anomaly
        }
    
    def __str__(self):
        """Return a string representation of the sensor reading."""
        return (f"Sensor {self.sensor_id} ({self.sensor_type} in {self.location}): "
                f"{self.value} at {datetime.fromtimestamp(self.timestamp).isoformat()}"
                f"{' (ANOMALY)' if self.is_anomaly else ''}")


class Sensor:
    """Base class for all sensor types."""
    
    def __init__(self, sensor_id, sensor_type, location):
        """
        Initialize a sensor.
        
        Args:
            sensor_id (str): Unique identifier for the sensor
            sensor_type (str): Type of sensor
            location (str): Location of the sensor in the home
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.location = location
        
        # Set normal operating parameters based on sensor type
        self._set_operating_parameters()
        
        # Last few readings for trend analysis
        self.history = []
        self.history_max_size = 10
        
        logger.info(f"Initialized {sensor_type} sensor {sensor_id} in {location}")
    
    def _set_operating_parameters(self):
        """Set normal operating parameters based on sensor type."""
        # Default parameters (will be overridden by subclasses)
        self.min_value = 0
        self.max_value = 100
        self.normal_mean = 50
        self.normal_std = 5
        self.anomaly_std_multiplier = 3
    
    def generate_reading(self, force_anomaly=False):
        """
        Generate a sensor reading.
        
        Args:
            force_anomaly (bool, optional): Whether to force an anomalous reading
            
        Returns:
            SensorReading: A sensor reading object
        """
        # Determine if this reading should be anomalous
        is_anomaly = force_anomaly or (random.random() < ANOMALY_PROBABILITY)
        
        # Generate a value based on whether this is an anomaly
        if is_anomaly:
            # For anomalies, we'll use a different distribution
            std_dev = self.normal_std * self.anomaly_std_multiplier
            value = random.gauss(self.normal_mean, std_dev)
            
            # Sometimes generate extreme values for more obvious anomalies
            if random.random() < 0.3:
                if random.random() < 0.5:
                    # High extreme
                    value = self.max_value * (1 + random.random() * 0.2)
                else:
                    # Low extreme
                    value = self.min_value * (1 - random.random() * 0.2)
        else:
            # Normal reading
            value = random.gauss(self.normal_mean, self.normal_std)
        
        # Ensure the value is within acceptable bounds
        value = max(self.min_value, min(self.max_value, value))
        
        # Create and return the reading
        reading = SensorReading(
            self.sensor_id,
            self.sensor_type,
            self.location,
            round(value, 2),
            is_anomaly=is_anomaly
        )
        
        # Add to history and maintain history size
        self.history.append(reading)
        if len(self.history) > self.history_max_size:
            self.history.pop(0)
        
        return reading


class TemperatureSensor(Sensor):
    """Temperature sensor that generates readings in Celsius."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to temperature sensors."""
        self.min_value = -10
        self.max_value = 50
        
        # Different normal mean based on location
        if self.location == 'kitchen':
            self.normal_mean = 23
        elif self.location == 'bathroom':
            self.normal_mean = 24
        elif self.location == 'garage':
            self.normal_mean = 18
        else:
            self.normal_mean = 21
        
        self.normal_std = 1.5
        self.anomaly_std_multiplier = 4


class HumiditySensor(Sensor):
    """Humidity sensor that generates readings as percentage."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to humidity sensors."""
        self.min_value = 0
        self.max_value = 100
        
        # Different normal mean based on location
        if self.location == 'bathroom':
            self.normal_mean = 65
        elif self.location == 'kitchen':
            self.normal_mean = 60
        else:
            self.normal_mean = 45
        
        self.normal_std = 5
        self.anomaly_std_multiplier = 3


class MotionSensor(Sensor):
    """Motion sensor that generates binary readings (0=no motion, 1=motion detected)."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to motion sensors."""
        self.min_value = 0
        self.max_value = 1
        
        # Probability of motion detection based on location and time of day
        self.motion_probability = 0.1  # Default
        self.anomaly_std_multiplier = 1  # Not used for binary sensors
    
    def generate_reading(self, force_anomaly=False):
        """Generate a binary motion reading."""
        # Time-based probability
        hour = datetime.now().hour
        
        # Higher probability during active hours (7am-11pm)
        if 7 <= hour <= 23:
            base_probability = 0.3
        else:
            base_probability = 0.05  # Lower at night
            
        # Location-based adjustments
        if self.location == 'entrance':
            base_probability *= 0.5  # Less frequent motion
        elif self.location == 'living_room':
            base_probability *= 1.5  # More frequent motion
        
        # Determine if motion is detected
        is_anomaly = force_anomaly or (random.random() < ANOMALY_PROBABILITY)
        
        if is_anomaly:
            # For anomalies, we'll invert the expected behavior
            if hour < 7 or hour > 23:  # Night time
                # Anomaly: unexpected motion at night
                value = 1 if random.random() < 0.8 else 0
            else:
                # Anomaly: unexpected lack of motion during day
                # or continuous motion (would need to be detected by the ML model)
                value = 1 if random.random() < 0.9 else 0
        else:
            # Normal behavior
            value = 1 if random.random() < base_probability else 0
        
        # Create and return the reading
        reading = SensorReading(
            self.sensor_id,
            self.sensor_type,
            self.location,
            value,
            is_anomaly=is_anomaly
        )
        
        # Add to history and maintain history size
        self.history.append(reading)
        if len(self.history) > self.history_max_size:
            self.history.pop(0)
        
        return reading


class SmokeDetector(Sensor):
    """Smoke detector that generates readings in ppm (parts per million)."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to smoke sensors."""
        self.min_value = 0
        self.max_value = 1000
        self.normal_mean = 5  # Normal background level
        self.normal_std = 2
        self.anomaly_std_multiplier = 10  # Big difference for smoke (fire) detection


class CO2Sensor(Sensor):
    """CO2 sensor that generates readings in ppm (parts per million)."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to CO2 sensors."""
        self.min_value = 300  # Outdoor-like levels
        self.max_value = 5000  # Hazardous levels
        
        # Different normal mean based on location
        if self.location == 'kitchen':
            self.normal_mean = 800
        elif self.location == 'bedroom':
            self.normal_mean = 600
        else:
            self.normal_mean = 700
        
        self.normal_std = 100
        self.anomaly_std_multiplier = 3


class DoorSensor(Sensor):
    """Door sensor that generates binary readings (0=closed, 1=open)."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to door sensors."""
        self.min_value = 0
        self.max_value = 1
        self.anomaly_std_multiplier = 1  # Not used for binary sensors
    
    def generate_reading(self, force_anomaly=False):
        """Generate a binary door reading."""
        # Time-based probability
        hour = datetime.now().hour
        
        # Determine the expected state based on time and location
        is_anomaly = force_anomaly or (random.random() < ANOMALY_PROBABILITY)
        
        # Normal behavior - mostly closed with occasional opening
        door_open_probability = 0.05  # Default
        
        # Entrance door is used more often
        if self.location == 'entrance':
            door_open_probability = 0.1
            # More likely to be open during common coming/going hours
            if hour in [7, 8, 9, 17, 18, 19]:
                door_open_probability = 0.2
        
        if is_anomaly:
            # Anomalies:
            # 1. Door left open at unusual hours
            # 2. Rapid open/close sequences (would need to be detected by ML)
            # 3. Door open in the middle of the night
            if hour < 6 or hour > 22:  # Night time
                value = 1  # Door open at night is suspicious
            else:
                value = 1 if random.random() < 0.8 else 0  # Mostly open during anomalies
        else:
            # Normal behavior
            value = 1 if random.random() < door_open_probability else 0
            
        # Create and return the reading
        reading = SensorReading(
            self.sensor_id,
            self.sensor_type,
            self.location,
            value,
            is_anomaly=is_anomaly
        )
        
        # Add to history and maintain history size
        self.history.append(reading)
        if len(self.history) > self.history_max_size:
            self.history.pop(0)
        
        return reading


class LightSensor(Sensor):
    """Light sensor that generates readings in lux."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to light sensors."""
        self.min_value = 0
        self.max_value = 10000  # Maximum daylight
        self.anomaly_std_multiplier = 3
        
        # Set the baseline according to location and time of day
        hour = datetime.now().hour
        
        # Nighttime
        if hour < 6 or hour > 21:
            self.normal_mean = 5  # Very low light at night
            self.normal_std = 3
        # Morning/Evening
        elif 6 <= hour < 9 or 18 <= hour <= 21:
            self.normal_mean = 200
            self.normal_std = 50
        # Daytime
        else:
            if self.location in ['living_room', 'kitchen']:
                self.normal_mean = 500  # Brighter areas
            else:
                self.normal_mean = 300
            self.normal_std = 100
    
    def generate_reading(self, force_anomaly=False):
        """Override to update parameters based on time before generating."""
        # Update parameters based on current time
        self._set_operating_parameters()
        # Then generate normally
        return super().generate_reading(force_anomaly)


class WindowSensor(Sensor):
    """Window sensor that generates binary readings (0=closed, 1=open)."""
    
    def _set_operating_parameters(self):
        """Set parameters specific to window sensors."""
        self.min_value = 0
        self.max_value = 1
        self.anomaly_std_multiplier = 1  # Not used for binary sensors
    
    def generate_reading(self, force_anomaly=False):
        """Generate a binary window reading."""
        # Time-based probability and weather would factor in here
        hour = datetime.now().hour
        
        # Determine the expected state based on time and location
        is_anomaly = force_anomaly or (random.random() < ANOMALY_PROBABILITY)
        
        # Normal behavior - windows mostly closed
        window_open_probability = 0.1  # Default
        
        # More likely to be open during nice-weather hours
        if 10 <= hour <= 18:
            window_open_probability = 0.3
        
        if is_anomaly:
            # Anomalies:
            # 1. Window open during rain/cold (would need weather API)
            # 2. Window open at unusual hours
            # 3. Window open when no one is home (would need occupancy data)
            if hour < 7 or hour > 22:  # Night time
                value = 1  # Window open at night might be suspicious
            else:
                value = 1 if random.random() < 0.7 else 0  # Mostly open during anomalies
        else:
            # Normal behavior
            value = 1 if random.random() < window_open_probability else 0
            
        # Create and return the reading
        reading = SensorReading(
            self.sensor_id,
            self.sensor_type,
            self.location,
            value,
            is_anomaly=is_anomaly
        )
        
        # Add to history and maintain history size
        self.history.append(reading)
        if len(self.history) > self.history_max_size:
            self.history.pop(0)
        
        return reading


class SensorManager:
    """Manages all sensors in the smart home system."""
    
    def __init__(self):
        """Initialize the sensor manager and create sensors."""
        self.sensors = {}
        self.data_file = os.path.join(DATA_DIR, 'sensor_data.jsonl')
        self.running = False
        self.simulation_thread = None
        
        # Dictionary mapping sensor types to classes
        self.sensor_classes = {
            'temperature': TemperatureSensor,
            'humidity': HumiditySensor,
            'motion': MotionSensor,
            'light': LightSensor,
            'door': DoorSensor,
            'window': WindowSensor,
            'smoke': SmokeDetector,
            'co2': CO2Sensor
        }
        
        # Create sensors for different locations and types
        self._create_sensors()
        
        logger.info(f"SensorManager initialized with {len(self.sensors)} sensors")
    
    def _create_sensors(self):
        """Create various sensors throughout the home."""
        # For each location, determine which sensor types make sense
        location_sensor_mapping = {
            'living_room': ['temperature', 'humidity', 'motion', 'light', 'window'],
            'kitchen': ['temperature', 'humidity', 'motion', 'smoke', 'co2', 'window'],
            'bedroom': ['temperature', 'humidity', 'motion', 'light', 'window'],
            'bathroom': ['temperature', 'humidity', 'motion', 'window'],
            'entrance': ['temperature', 'motion', 'door'],
            'garage': ['temperature', 'humidity', 'motion', 'door', 'co2']
        }
        
        # Create appropriate sensors for each location
        for location, sensor_types in location_sensor_mapping.items():
            for sensor_type in sensor_types:
                sensor_id = f"{location}_{sensor_type}"
                sensor_class = self.sensor_classes[sensor_type]
                self.sensors[sensor_id] = sensor_class(sensor_id, sensor_type, location)
    
    def get_all_sensors(self):
        """Return a list of all sensors."""
        return list(self.sensors.values())
    
    def get_sensor(self, sensor_id):
        """Get a specific sensor by ID."""
        return self.sensors.get(sensor_id)
    
    def get_sensors_by_type(self, sensor_type):
        """Get all sensors of a specific type."""
        return [s for s in self.sensors.values() if s.sensor_type == sensor_type]
    
    def get_sensors_by_location(self, location):
        """Get all sensors in a specific location."""
        return [s for s in self.sensors.values() if s.location == location]
    
    def get_reading(self, sensor_id, force_anomaly=False):
        """Get a reading from a specific sensor."""
        sensor = self.get_sensor(sensor_id)
        if sensor:
            return sensor.generate_reading(force_anomaly)
        return None
    
    def get_all_readings(self, force_anomaly_percentage=None):
        """
        Get readings from all sensors.
        
        Args:
            force_anomaly_percentage (float, optional): Percentage of sensors to force anomalies for
                
        Returns:
            list: List of SensorReading objects
        """
        readings = []
        
        # If forcing anomalies for a percentage of sensors, determine which ones
        anomaly_sensors = []
        if force_anomaly_percentage is not None:
            num_anomaly_sensors = int(len(self.sensors) * force_anomaly_percentage)
            anomaly_sensors = random.sample(list(self.sensors.keys()), num_anomaly_sensors)
        
        # Generate readings for all sensors
        for sensor_id, sensor in self.sensors.items():
            force_anomaly = sensor_id in anomaly_sensors
            readings.append(sensor.generate_reading(force_anomaly))
        
        return readings
    
    def save_readings_to_file(self, readings):
        """
        Save sensor readings to a JSON Lines file.
        
        Args:
            readings (list): List of SensorReading objects
        """
        with open(self.data_file, 'a') as f:
            for reading in readings:
                f.write(json.dumps(reading.to_dict()) + '\n')
    
    def start_simulation(self, callback=None):
        """
        Start the sensor simulation.
        
        Args:
            callback (callable, optional): Function to call with each batch of readings
        """
        if self.running:
            logger.warning("Simulation already running")
            return
        
        self.running = True
        self.simulation_thread = threading.Thread(
            target=self._run_simulation,
            args=(callback,),
            daemon=True
        )
        self.simulation_thread.start()
        logger.info("Sensor simulation started")
    
    def stop_simulation(self):
        """Stop the sensor simulation."""
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5)
        logger.info("Sensor simulation stopped")
    
    def _run_simulation(self, callback=None):
        """
        Run the sensor simulation loop.
        
        Args:
            callback (callable, optional): Function to call with each batch of readings
        """
        while self.running:
            try:
                # Get readings from all sensors
                readings = self.get_all_readings()
                
                # Save readings to file
                self.save_readings_to_file(readings)
                
                # Call the callback function if provided
                if callback:
                    callback(readings)
                
                # Wait for the next update interval
                time.sleep(SENSOR_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in sensor simulation: {e}")
                time.sleep(1)  # Sleep briefly before retrying


# Example usage
if __name__ == "__main__":
    # Create the sensor manager
    manager = SensorManager()
    
    # Define a callback function to process readings
    def process_readings(readings):
        """Process a batch of sensor readings."""
        for reading in readings:
            if reading.is_anomaly:
                logger.warning(f"Anomaly detected: {reading}")
            else:
                logger.debug(f"Normal reading: {reading}")
    
    # Start the simulation
    manager.start_simulation(callback=process_readings)
    
    try:
        # Run for a while
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Stop the simulation
        manager.stop_simulation()
