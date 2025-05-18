"""
Tests for the anomaly detector
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.anomaly_detector import (
    AnomalyDetectionModel, IsolationForestModel, 
    LocalOutlierFactorModel, LSTMAutoencoder, AnomalyDetector
)

class TestIsolationForestModel(unittest.TestCase):
    """Tests for the IsolationForestModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the MODELS_DIR constant
        self.original_models_dir = sys.modules['src.ml.anomaly_detector'].MODELS_DIR
        sys.modules['src.ml.anomaly_detector'].MODELS_DIR = self.temp_dir
        
        # Create a test model
        self.model = IsolationForestModel(
            sensor_type="temperature",
            location="living_room",
            n_estimators=10,
            contamination=0.1
        )
        
        # Create test data
        np.random.seed(42)
        self.normal_data = pd.DataFrame({
            'value': np.random.normal(21, 1, 100),
            'hour': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100)
        })
        
        # Add anomalies
        self.anomalous_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(21, 1, 90),  # Normal values
                np.random.normal(30, 2, 10)   # Anomalous values (higher temperature)
            ]),
            'hour': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100)
        })
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Restore the original MODELS_DIR
        sys.modules['src.ml.anomaly_detector'].MODELS_DIR = self.original_models_dir
    
    def test_model_initialization(self):
        """Test initializing the model."""
        self.assertEqual(self.model.sensor_type, "temperature")
        self.assertEqual(self.model.location, "living_room")
        self.assertEqual(self.model.n_estimators, 10)
        self.assertEqual(self.model.contamination, 0.1)
        self.assertFalse(self.model.is_trained)
    
    def test_model_training(self):
        """Test training the model."""
        result = self.model.train(self.normal_data)
        
        self.assertTrue(result)
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
    
    def test_model_prediction(self):
        """Test making predictions with the model."""
        self.model.train(self.normal_data)
        
        scores = self.model.predict(self.anomalous_data)
        
        self.assertEqual(len(scores), len(self.anomalous_data))
        
        # The last 10 samples should have higher anomaly scores
        mean_normal_score = np.mean(scores[:90])
        mean_anomaly_score = np.mean(scores[90:])
        
        self.assertLess(mean_normal_score, mean_anomaly_score)
    
    def test_model_anomaly_detection(self):
        """Test detecting anomalies with the model."""
        self.model.train(self.normal_data)
        
        result = self.model.detect_anomalies(self.anomalous_data, threshold=1.5)
        
        self.assertIn('anomaly_score', result.columns)
        self.assertIn('is_anomaly', result.columns)
        
        # There should be some anomalies detected
        self.assertTrue(result['is_anomaly'].sum() > 0)
        
        # There should be more anomalies in the last 10 samples
        anomaly_counts = result['is_anomaly'].iloc[90:].sum()
        self.assertGreater(anomaly_counts, 0)
    
    def test_model_save_load(self):
        """Test saving and loading the model."""
        self.model.train(self.normal_data)
        
        # Save the model
        save_result = self.model.save()
        self.assertTrue(save_result)
        
        # Create a new model
        new_model = IsolationForestModel(
            sensor_type="temperature",
            location="living_room"
        )
        
        # Load the saved model
        load_result = new_model.load()
        self.assertTrue(load_result)
        self.assertTrue(new_model.is_trained)
        
        # Make predictions with both models
        original_scores = self.model.predict(self.anomalous_data)
        loaded_scores = new_model.predict(self.anomalous_data)
        
        # Predictions should be the same
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)


class TestAnomalyDetector(unittest.TestCase):
    """Tests for the AnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the MODELS_DIR constant
        self.original_models_dir = sys.modules['src.ml.anomaly_detector'].MODELS_DIR
        sys.modules['src.ml.anomaly_detector'].MODELS_DIR = self.temp_dir
        
        # Also mock the DATA_DIR constant
        self.original_data_dir = sys.modules['src.ml.anomaly_detector'].DATA_DIR
        sys.modules['src.ml.anomaly_detector'].DATA_DIR = self.temp_dir
        
        # Create a test detector
        self.detector = AnomalyDetector()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Restore the original constants
        sys.modules['src.ml.anomaly_detector'].MODELS_DIR = self.original_models_dir
        sys.modules['src.ml.anomaly_detector'].DATA_DIR = self.original_data_dir
    
    def test_detector_initialization(self):
        """Test initializing the detector."""
        self.assertIsNotNone(self.detector.models)
        self.assertGreater(len(self.detector.models), 0)
    
    def test_get_model(self):
        """Test getting a specific model."""
        model = self.detector.get_model('isolation_forest', 'temperature', 'living_room')
        
        self.assertIsNotNone(model)
        self.assertIsInstance(model, IsolationForestModel)
        self.assertEqual(model.sensor_type, 'temperature')
        self.assertEqual(model.location, 'living_room')
    
    def test_detect_anomalies_without_training(self):
        """Test detecting anomalies without training."""
        # Create some test readings
        readings = [
            {
                'sensor_id': 'living_room_temperature',
                'sensor_type': 'temperature',
                'location': 'living_room',
                'value': 22.5,
                'timestamp': 1620000000.0,
                'datetime': '2021-05-03T00:00:00',
                'is_anomaly': False
            },
            {
                'sensor_id': 'kitchen_temperature',
                'sensor_type': 'temperature',
                'location': 'kitchen',
                'value': 35.0,  # Anomalously high
                'timestamp': 1620000000.0,
                'datetime': '2021-05-03T00:00:00',
                'is_anomaly': True
            }
        ]
        
        results = self.detector.detect_anomalies(readings)
        
        self.assertEqual(len(results), len(readings))
        self.assertTrue(any(r.get('is_anomaly', False) for r in results))


# We'll skip testing LSTMAutoencoder in detail as it requires TensorFlow
# and would slow down the tests significantly. A placeholder test is included:
class TestLSTMAutoencoder(unittest.TestCase):
    """Simple test for LSTMAutoencoder class."""
    
    def test_model_initialization(self):
        """Test initializing the model."""
        model = LSTMAutoencoder(
            sensor_type="temperature",
            location="living_room",
            seq_length=5,
            lstm_units=32,
            epochs=5,
            batch_size=16
        )
        
        self.assertEqual(model.sensor_type, "temperature")
        self.assertEqual(model.location, "living_room")
        self.assertEqual(model.seq_length, 5)
        self.assertEqual(model.lstm_units, 32)
        self.assertEqual(model.epochs, 5)
        self.assertEqual(model.batch_size, 16)
        self.assertFalse(model.is_trained)


if __name__ == "__main__":
    unittest.main()
