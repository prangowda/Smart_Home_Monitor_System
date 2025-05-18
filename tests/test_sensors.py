"""
Tests for the sensor simulator
"""
import sys
import os
import unittest
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.sensor_simulator import (
    SensorReading, Sensor, TemperatureSensor, HumiditySensor,
    MotionSensor, SensorManager
)

class TestSensorReading(unittest.TestCase):
    """Tests for the SensorReading class."""
    
    def test_sensor_reading_creation(self):
        """Test creating a sensor reading."""
        reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type="temperature",
            location="living_room",
            value=23.5,
            is_anomaly=False
        )
        
        self.assertEqual(reading.sensor_id, "test_sensor")
        self.assertEqual(reading.sensor_type, "temperature")
        self.assertEqual(reading.location, "living_room")
        self.assertEqual(reading.value, 23.5)
        self.assertFalse(reading.is_anomaly)
    
    def test_sensor_reading_to_dict(self):
        """Test converting a sensor reading to a dictionary."""
        timestamp = 1620000000.0
        reading = SensorReading(
            sensor_id="test_sensor",
            sensor_type="temperature",
            location="living_room",
            value=23.5,
            timestamp=timestamp,
            is_anomaly=True
        )
        
        reading_dict = reading.to_dict()
        
        self.assertEqual(reading_dict["sensor_id"], "test_sensor")
        self.assertEqual(reading_dict["sensor_type"], "temperature")
        self.assertEqual(reading_dict["location"], "living_room")
        self.assertEqual(reading_dict["value"], 23.5)
        self.assertEqual(reading_dict["timestamp"], timestamp)
        self.assertEqual(reading_dict["datetime"], datetime.fromtimestamp(timestamp).isoformat())
        self.assertTrue(reading_dict["is_anomaly"])


class TestSensor(unittest.TestCase):
    """Tests for the Sensor base class."""
    
    def test_sensor_initialization(self):
        """Test initializing a sensor."""
        sensor = Sensor(
            sensor_id="test_sensor",
            sensor_type="generic",
            location="living_room"
        )
        
        self.assertEqual(sensor.sensor_id, "test_sensor")
        self.assertEqual(sensor.sensor_type, "generic")
        self.assertEqual(sensor.location, "living_room")
        self.assertEqual(len(sensor.history), 0)
    
    def test_sensor_reading_generation(self):
        """Test generating a sensor reading."""
        sensor = Sensor(
            sensor_id="test_sensor",
            sensor_type="generic",
            location="living_room"
        )
        
        reading = sensor.generate_reading()
        
        self.assertEqual(reading.sensor_id, "test_sensor")
        self.assertEqual(reading.sensor_type, "generic")
        self.assertEqual(reading.location, "living_room")
        self.assertFalse(reading.is_anomaly)
        
        # Test history tracking
        self.assertEqual(len(sensor.history), 1)
        self.assertEqual(sensor.history[0], reading)
    
    def test_forced_anomaly(self):
        """Test generating an anomalous reading."""
        sensor = Sensor(
            sensor_id="test_sensor",
            sensor_type="generic",
            location="living_room"
        )
        
        reading = sensor.generate_reading(force_anomaly=True)
        
        self.assertTrue(reading.is_anomaly)


class TestTemperatureSensor(unittest.TestCase):
    """Tests for the TemperatureSensor class."""
    
    def test_temperature_sensor_initialization(self):
        """Test initializing a temperature sensor."""
        sensor = TemperatureSensor(
            sensor_id="temp_sensor",
            sensor_type="temperature",
            location="living_room"
        )
        
        self.assertEqual(sensor.sensor_id, "temp_sensor")
        self.assertEqual(sensor.sensor_type, "temperature")
        self.assertEqual(sensor.location, "living_room")
        self.assertGreater(sensor.normal_mean, -10)
        self.assertLess(sensor.normal_mean, 50)
    
    def test_temperature_reading_generation(self):
        """Test generating a temperature reading."""
        sensor = TemperatureSensor(
            sensor_id="temp_sensor",
            sensor_type="temperature",
            location="living_room"
        )
        
        reading = sensor.generate_reading()
        
        self.assertEqual(reading.sensor_id, "temp_sensor")
        self.assertEqual(reading.sensor_type, "temperature")
        self.assertEqual(reading.location, "living_room")
        self.assertGreaterEqual(reading.value, sensor.min_value)
        self.assertLessEqual(reading.value, sensor.max_value)


class TestSensorManager(unittest.TestCase):
    """Tests for the SensorManager class."""
    
    def test_sensor_manager_initialization(self):
        """Test initializing the sensor manager."""
        manager = SensorManager()
        
        self.assertGreater(len(manager.sensors), 0)
    
    def test_get_all_sensors(self):
        """Test getting all sensors."""
        manager = SensorManager()
        
        sensors = manager.get_all_sensors()
        
        self.assertGreater(len(sensors), 0)
        self.assertTrue(all(isinstance(s, Sensor) for s in sensors))
    
    def test_get_sensor(self):
        """Test getting a specific sensor."""
        manager = SensorManager()
        
        # Get the first sensor ID
        sensor_id = list(manager.sensors.keys())[0]
        
        sensor = manager.get_sensor(sensor_id)
        
        self.assertIsNotNone(sensor)
        self.assertEqual(sensor.sensor_id, sensor_id)
    
    def test_get_all_readings(self):
        """Test getting readings from all sensors."""
        manager = SensorManager()
        
        readings = manager.get_all_readings()
        
        self.assertEqual(len(readings), len(manager.sensors))
        self.assertTrue(all(isinstance(r, SensorReading) for r in readings))
    
    def test_get_sensors_by_type(self):
        """Test getting sensors by type."""
        manager = SensorManager()
        
        # Get temperature sensors
        temp_sensors = manager.get_sensors_by_type("temperature")
        
        self.assertGreater(len(temp_sensors), 0)
        self.assertTrue(all(s.sensor_type == "temperature" for s in temp_sensors))
    
    def test_get_sensors_by_location(self):
        """Test getting sensors by location."""
        manager = SensorManager()
        
        # Get living room sensors
        living_room_sensors = manager.get_sensors_by_location("living_room")
        
        self.assertGreater(len(living_room_sensors), 0)
        self.assertTrue(all(s.location == "living_room" for s in living_room_sensors))


if __name__ == "__main__":
    unittest.main()
