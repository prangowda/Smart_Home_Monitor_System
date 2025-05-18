"""
Anomaly Detection for Smart Home Monitoring System

This module implements machine learning algorithms for detecting anomalies
in sensor data, using various techniques including Isolation Forest,
Local Outlier Factor, and an LSTM Autoencoder.
"""
import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SENSOR_TYPES, SENSOR_LOCATIONS, ANOMALY_THRESHOLD,
    MODEL_UPDATE_INTERVAL, TRAINING_DATA_WINDOW,
    DATA_DIR, MODELS_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('anomaly_detector')

class AnomalyDetectionModel:
    """Base class for anomaly detection models."""
    
    def __init__(self, model_name, sensor_type=None, location=None):
        """
        Initialize the anomaly detection model.
        
        Args:
            model_name (str): Name of the model
            sensor_type (str, optional): Type of sensor this model is for
            location (str, optional): Location this model is for
        """
        self.model_name = model_name
        self.sensor_type = sensor_type
        self.location = location
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        # Define model path
        self.model_path = self._get_model_path()
        
        # Try to load existing model
        self.load()
        
        logger.info(f"Initialized {model_name} model for {sensor_type} in {location}")
    
    def _get_model_path(self):
        """Get the path to save/load the model."""
        if self.sensor_type and self.location:
            return os.path.join(MODELS_DIR, f"{self.model_name}_{self.sensor_type}_{self.location}.pkl")
        elif self.sensor_type:
            return os.path.join(MODELS_DIR, f"{self.model_name}_{self.sensor_type}.pkl")
        else:
            return os.path.join(MODELS_DIR, f"{self.model_name}.pkl")
    
    def save(self):
        """Save the model to disk."""
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self):
        """Load the model from disk."""
        raise NotImplementedError("Subclasses must implement load()")
    
    def train(self, data):
        """Train the model on the provided data."""
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, data):
        """Make predictions with the model."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def detect_anomalies(self, data, threshold=None):
        """
        Detect anomalies in the data.
        
        Args:
            data (pandas.DataFrame): Data to detect anomalies in
            threshold (float, optional): Threshold for anomaly detection
            
        Returns:
            pandas.DataFrame: Data with anomaly scores and flags
        """
        raise NotImplementedError("Subclasses must implement detect_anomalies()")


class IsolationForestModel(AnomalyDetectionModel):
    """Anomaly detection using Isolation Forest algorithm."""
    
    def __init__(self, sensor_type=None, location=None, n_estimators=100, contamination=0.05):
        """
        Initialize the Isolation Forest model.
        
        Args:
            sensor_type (str, optional): Type of sensor this model is for
            location (str, optional): Location this model is for
            n_estimators (int, optional): Number of estimators in the forest
            contamination (float, optional): Expected proportion of anomalies
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        super().__init__("isolation_forest", sensor_type, location)
    
    def save(self):
        """Save the model to disk using joblib."""
        if self.model and self.is_trained:
            import joblib
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
            logger.info(f"Saved model to {self.model_path}")
            return True
        logger.warning("Model not trained, nothing to save")
        return False
    
    def load(self):
        """Load the model from disk using joblib."""
        if os.path.exists(self.model_path):
            try:
                import joblib
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
                logger.info(f"Loaded model from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        return False
    
    def train(self, data):
        """
        Train the Isolation Forest model.
        
        Args:
            data (pandas.DataFrame): Training data
            
        Returns:
            bool: Whether training was successful
        """
        if data.empty:
            logger.warning("Empty training data provided")
            return False
        
        try:
            # Scale the data
            X = self.scaler.fit_transform(data)
            
            # Create and fit the model
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(X)
            
            self.is_trained = True
            logger.info(f"Trained Isolation Forest model on {len(data)} samples")
            
            # Save the model
            self.save()
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, data):
        """
        Make predictions with the Isolation Forest model.
        
        Args:
            data (pandas.DataFrame): Data to make predictions on
            
        Returns:
            numpy.ndarray: Anomaly scores
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot make predictions")
            return None
        
        try:
            # Scale the data
            X = self.scaler.transform(data)
            
            # Get decision function values (negative of anomaly scores)
            scores = self.model.decision_function(X)
            
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = -scores
            
            return anomaly_scores
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def detect_anomalies(self, data, threshold=None):
        """
        Detect anomalies in the data using Isolation Forest.
        
        Args:
            data (pandas.DataFrame): Data to detect anomalies in
            threshold (float, optional): Threshold for anomaly detection
            
        Returns:
            pandas.DataFrame: Data with anomaly scores and flags
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot detect anomalies")
            return data
        
        threshold = threshold or ANOMALY_THRESHOLD
        
        try:
            # Get anomaly scores
            anomaly_scores = self.predict(data)
            
            if anomaly_scores is None:
                return data
            
            # Add scores to the dataframe
            result = data.copy()
            result['anomaly_score'] = anomaly_scores
            
            # Determine anomalies
            result['is_anomaly'] = result['anomaly_score'] > threshold
            
            return result
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return data


class LSTMAutoencoder(AnomalyDetectionModel):
    """Anomaly detection using LSTM Autoencoder neural network."""
    
    def __init__(self, sensor_type=None, location=None, seq_length=10, 
                 lstm_units=64, epochs=50, batch_size=32):
        """
        Initialize the LSTM Autoencoder model.
        
        Args:
            sensor_type (str, optional): Type of sensor this model is for
            location (str, optional): Location this model is for
            seq_length (int, optional): Length of input sequences
            lstm_units (int, optional): Number of LSTM units
            epochs (int, optional): Number of training epochs
            batch_size (int, optional): Batch size for training
        """
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.reconstruction_error_threshold = None
        
        # TensorFlow settings
        tf.keras.backend.clear_session()
        
        super().__init__("lstm_autoencoder", sensor_type, location)
    
    def _build_model(self, input_dim):
        """
        Build the LSTM Autoencoder model.
        
        Args:
            input_dim (int): Dimensionality of the input data
            
        Returns:
            keras.Model: The LSTM Autoencoder model
        """
        # Define model architecture
        inputs = Input(shape=(self.seq_length, input_dim))
        
        # Encoder
        encoded = LSTM(self.lstm_units, activation='relu')(inputs)
        
        # Bottleneck
        bottleneck = RepeatVector(self.seq_length)(encoded)
        
        # Decoder
        decoded = LSTM(self.lstm_units, activation='relu', return_sequences=True)(bottleneck)
        outputs = TimeDistributed(Dense(input_dim))(decoded)
        
        # Create and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def _create_sequences(self, data, seq_length):
        """
        Create sequences from the data for LSTM training.
        
        Args:
            data (numpy.ndarray): Input data
            seq_length (int): Length of sequences
            
        Returns:
            numpy.ndarray: Sequences for LSTM training
        """
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    def save(self):
        """Save the model to disk."""
        if self.model and self.is_trained:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save Keras model (replace .pkl with .h5)
            model_path = self.model_path.replace('.pkl', '.h5')
            self.model.save(model_path)
            
            # Save scaler and other metadata
            import joblib
            metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
            joblib.dump({
                'scaler': self.scaler,
                'reconstruction_error_threshold': self.reconstruction_error_threshold,
                'seq_length': self.seq_length,
                'input_dim': self.model.input_shape[-1]
            }, metadata_path)
            
            logger.info(f"Saved model to {model_path} and metadata to {metadata_path}")
            return True
        
        logger.warning("Model not trained, nothing to save")
        return False
    
    def load(self):
        """Load the model from disk."""
        # Check for model file (with .h5 extension)
        model_path = self.model_path.replace('.pkl', '.h5')
        metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            try:
                # Load the Keras model
                self.model = load_model(model_path)
                
                # Load metadata
                import joblib
                metadata = joblib.load(metadata_path)
                self.scaler = metadata['scaler']
                self.reconstruction_error_threshold = metadata['reconstruction_error_threshold']
                self.seq_length = metadata['seq_length']
                
                self.is_trained = True
                logger.info(f"Loaded model from {model_path} and metadata from {metadata_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        return False
    
    def train(self, data):
        """
        Train the LSTM Autoencoder model.
        
        Args:
            data (pandas.DataFrame): Training data
            
        Returns:
            bool: Whether training was successful
        """
        if data.empty:
            logger.warning("Empty training data provided")
            return False
        
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            sequences = self._create_sequences(scaled_data, self.seq_length)
            
            if len(sequences) == 0:
                logger.warning("Not enough data to create sequences")
                return False
            
            # Build the model
            input_dim = data.shape[1]
            self.model = self._build_model(input_dim)
            
            # Train the model
            self.model.fit(
                sequences, sequences,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=1
            )
            
            # Calculate reconstruction error threshold
            reconstructions = self.model.predict(sequences)
            mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
            # Set threshold at 95th percentile
            self.reconstruction_error_threshold = np.percentile(mse, 95)
            
            self.is_trained = True
            logger.info(f"Trained LSTM Autoencoder model on {len(sequences)} sequences")
            
            # Save the model
            self.save()
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, data):
        """
        Make predictions with the LSTM Autoencoder model.
        
        Args:
            data (pandas.DataFrame): Data to make predictions on
            
        Returns:
            numpy.ndarray: Reconstruction errors (anomaly scores)
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot make predictions")
            return None
        
        try:
            # Scale the data
            scaled_data = self.scaler.transform(data)
            
            # Create sequences
            sequences = self._create_sequences(scaled_data, self.seq_length)
            
            if len(sequences) == 0:
                logger.warning("Not enough data to create sequences")
                return None
            
            # Get reconstructions
            reconstructions = self.model.predict(sequences)
            
            # Calculate reconstruction errors (MSE)
            mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
            
            # Map the sequence errors back to original data points by using the max error
            # of all sequences containing each point
            anomaly_scores = np.zeros(len(data))
            for i in range(len(mse)):
                idx_range = range(i, min(i + self.seq_length, len(data)))
                anomaly_scores[idx_range] = np.maximum(
                    anomaly_scores[idx_range], mse[i]
                )
            
            return anomaly_scores
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def detect_anomalies(self, data, threshold=None):
        """
        Detect anomalies in the data using LSTM Autoencoder.
        
        Args:
            data (pandas.DataFrame): Data to detect anomalies in
            threshold (float, optional): Threshold for anomaly detection
            
        Returns:
            pandas.DataFrame: Data with anomaly scores and flags
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot detect anomalies")
            return data
        
        threshold = threshold or self.reconstruction_error_threshold
        
        try:
            # Get anomaly scores
            anomaly_scores = self.predict(data)
            
            if anomaly_scores is None:
                return data
            
            # Add scores to the dataframe
            result = data.copy()
            result['anomaly_score'] = anomaly_scores
            
            # Determine anomalies
            result['is_anomaly'] = result['anomaly_score'] > threshold
            
            return result
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return data


class LocalOutlierFactorModel(AnomalyDetectionModel):
    """Anomaly detection using Local Outlier Factor algorithm."""
    
    def __init__(self, sensor_type=None, location=None, n_neighbors=20, contamination=0.05):
        """
        Initialize the Local Outlier Factor model.
        
        Args:
            sensor_type (str, optional): Type of sensor this model is for
            location (str, optional): Location this model is for
            n_neighbors (int, optional): Number of neighbors to use
            contamination (float, optional): Expected proportion of anomalies
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        super().__init__("local_outlier_factor", sensor_type, location)
    
    def save(self):
        """Save the model to disk using joblib."""
        if self.model and self.is_trained:
            import joblib
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
            logger.info(f"Saved model to {self.model_path}")
            return True
        logger.warning("Model not trained, nothing to save")
        return False
    
    def load(self):
        """Load the model from disk using joblib."""
        if os.path.exists(self.model_path):
            try:
                import joblib
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
                logger.info(f"Loaded model from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        return False
    
    def train(self, data):
        """
        Train the Local Outlier Factor model.
        
        Args:
            data (pandas.DataFrame): Training data
            
        Returns:
            bool: Whether training was successful
        """
        if data.empty:
            logger.warning("Empty training data provided")
            return False
        
        try:
            # Scale the data
            X = self.scaler.fit_transform(data)
            
            # Create and fit the model
            self.model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True  # Allow prediction on new data
            )
            self.model.fit(X)
            
            self.is_trained = True
            logger.info(f"Trained Local Outlier Factor model on {len(data)} samples")
            
            # Save the model
            self.save()
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, data):
        """
        Make predictions with the Local Outlier Factor model.
        
        Args:
            data (pandas.DataFrame): Data to make predictions on
            
        Returns:
            numpy.ndarray: Anomaly scores (negative of outlier factors)
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot make predictions")
            return None
        
        try:
            # Scale the data
            X = self.scaler.transform(data)
            
            # Get decision function values (negative of anomaly scores)
            scores = -self.model.decision_function(X)
            
            return scores
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def detect_anomalies(self, data, threshold=None):
        """
        Detect anomalies in the data using Local Outlier Factor.
        
        Args:
            data (pandas.DataFrame): Data to detect anomalies in
            threshold (float, optional): Threshold for anomaly detection
            
        Returns:
            pandas.DataFrame: Data with anomaly scores and flags
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot detect anomalies")
            return data
        
        threshold = threshold or ANOMALY_THRESHOLD
        
        try:
            # Get anomaly scores
            anomaly_scores = self.predict(data)
            
            if anomaly_scores is None:
                return data
            
            # Add scores to the dataframe
            result = data.copy()
            result['anomaly_score'] = anomaly_scores
            
            # Determine anomalies
            result['is_anomaly'] = result['anomaly_score'] > threshold
            
            return result
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return data


class AnomalyDetector:
    """Manager class for all anomaly detection models."""
    
    def __init__(self):
        """
        Initialize the anomaly detector with models for each sensor type and location.
        """
        self.models = {}
        self.running = False
        self.training_thread = None
        self.sensor_data_path = os.path.join(DATA_DIR, 'sensor_data.jsonl')
        
        logger.info("Initializing AnomalyDetector")
        
        # Initialize models
        self._initialize_models()
    
    def _get_model_key(self, model_type, sensor_type, location=None):
        """
        Get a unique key for a model.
        
        Args:
            model_type (str): Type of model (isolation_forest, lstm_autoencoder, etc.)
            sensor_type (str): Type of sensor
            location (str, optional): Location
            
        Returns:
            str: Unique key for the model
        """
        if location:
            return f"{model_type}_{sensor_type}_{location}"
        return f"{model_type}_{sensor_type}"
    
    def _initialize_models(self):
        """
        Initialize models for each sensor type and location.
        """
        # Create models for each sensor type
        for sensor_type in SENSOR_TYPES:
            # Create a general model for this sensor type
            self._create_models_for_sensor(sensor_type)
            
            # Create models for each location for this sensor type
            for location in SENSOR_LOCATIONS:
                self._create_models_for_sensor(sensor_type, location)
    
    def _create_models_for_sensor(self, sensor_type, location=None):
        """
        Create models for a specific sensor type and location.
        
        Args:
            sensor_type (str): Type of sensor
            location (str, optional): Location
        """
        # Isolation Forest model
        if_key = self._get_model_key('isolation_forest', sensor_type, location)
        self.models[if_key] = IsolationForestModel(sensor_type, location)
        
        # LSTM Autoencoder model (mainly for time-series sensors)
        if sensor_type in ['temperature', 'humidity', 'co2']:
            lstm_key = self._get_model_key('lstm_autoencoder', sensor_type, location)
            self.models[lstm_key] = LSTMAutoencoder(sensor_type, location)
        
        # Local Outlier Factor model
        lof_key = self._get_model_key('local_outlier_factor', sensor_type, location)
        self.models[lof_key] = LocalOutlierFactorModel(sensor_type, location)
    
    def get_model(self, model_type, sensor_type, location=None):
        """
        Get a specific model.
        
        Args:
            model_type (str): Type of model
            sensor_type (str): Type of sensor
            location (str, optional): Location
            
        Returns:
            AnomalyDetectionModel: The requested model
        """
        key = self._get_model_key(model_type, sensor_type, location)
        return self.models.get(key)
    
    def load_sensor_data(self, window_seconds=None):
        """
        Load sensor data from the data file.
        
        Args:
            window_seconds (int, optional): How many seconds of data to load
            
        Returns:
            dict: Dictionary mapping sensor types to pandas DataFrames
        """
        data = {}
        
        # Check if the data file exists
        if not os.path.exists(self.sensor_data_path):
            logger.warning(f"Data file {self.sensor_data_path} does not exist")
            return data
        
        try:
            # Load all data from the file
            all_readings = []
            with open(self.sensor_data_path, 'r') as f:
                for line in f:
                    all_readings.append(json.loads(line.strip()))
            
            # Filter by time window if specified
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                all_readings = [r for r in all_readings if r['timestamp'] >= cutoff_time]
            
            # Group by sensor type
            sensor_types = set(r['sensor_type'] for r in all_readings)
            
            for sensor_type in sensor_types:
                # Filter readings for this sensor type
                readings = [r for r in all_readings if r['sensor_type'] == sensor_type]
                
                # Convert to DataFrame
                df = pd.DataFrame(readings)
                
                # Prepare features for ML
                features = ['value']
                
                # Add time-based features
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
                features.extend(['hour', 'day_of_week'])
                
                # One-hot encode location
                location_dummies = pd.get_dummies(df['location'], prefix='location')
                df = pd.concat([df, location_dummies], axis=1)
                features.extend(location_dummies.columns.tolist())
                
                # Store the DataFrame with its feature columns
                data[sensor_type] = {
                    'df': df,
                    'features': features
                }
            
            logger.info(f"Loaded {len(all_readings)} sensor readings for {len(sensor_types)} sensor types")
            return data
        except Exception as e:
            logger.error(f"Error loading sensor data: {e}")
            return {}
    
    def train_models(self, force=False):
        """
        Train all models on the available sensor data.
        
        Args:
            force (bool, optional): Whether to force training even if models are already trained
            
        Returns:
            bool: Whether training was initiated
        """
        if self.running and not force:
            logger.warning("Training already in progress")
            return False
        
        # Start training in a separate thread
        self.running = True
        self.training_thread = threading.Thread(
            target=self._train_models_thread,
            daemon=True
        )
        self.training_thread.start()
        
        return True
    
    def _train_models_thread(self):
        """
        Thread function for training models.
        """
        try:
            # Load sensor data
            data = self.load_sensor_data(window_seconds=TRAINING_DATA_WINDOW)
            
            if not data:
                logger.warning("No data available for training")
                self.running = False
                return
            
            # Train models for each sensor type
            for sensor_type, sensor_data in data.items():
                self._train_models_for_sensor_type(sensor_type, sensor_data)
                
            logger.info("Finished training all models")
        except Exception as e:
            logger.error(f"Error in training thread: {e}")
        finally:
            self.running = False
    
    def _train_models_for_sensor_type(self, sensor_type, sensor_data):
        """
        Train models for a specific sensor type.
        
        Args:
            sensor_type (str): Type of sensor
            sensor_data (dict): Dictionary with dataframe and feature columns
        """
        df = sensor_data['df']
        features = sensor_data['features']
        
        # Extract feature data
        X = df[features]
        
        # Train general models for this sensor type
        for model_type in ['isolation_forest', 'local_outlier_factor']:
            model = self.get_model(model_type, sensor_type)
            if model:
                model.train(X)
        
        # Train LSTM model if applicable
        if sensor_type in ['temperature', 'humidity', 'co2']:
            lstm_model = self.get_model('lstm_autoencoder', sensor_type)
            if lstm_model:
                lstm_model.train(X)
        
        # Train location-specific models
        locations = df['location'].unique()
        for location in locations:
            # Filter data for this location
            location_df = df[df['location'] == location]
            if len(location_df) < 100:  # Skip if not enough data
                continue
                
            location_X = location_df[features]
            
            # Train models for this sensor type and location
            for model_type in ['isolation_forest', 'local_outlier_factor']:
                model = self.get_model(model_type, sensor_type, location)
                if model:
                    model.train(location_X)
            
            # Train LSTM model if applicable
            if sensor_type in ['temperature', 'humidity', 'co2']:
                lstm_model = self.get_model('lstm_autoencoder', sensor_type, location)
                if lstm_model:
                    lstm_model.train(location_X)
    
    def detect_anomalies(self, readings):
        """
        Detect anomalies in a batch of sensor readings.
        
        Args:
            readings (list): List of SensorReading objects or dictionaries
            
        Returns:
            list: The input readings with anomaly detection results added
        """
        if not readings:
            return []
        
        # Group readings by sensor type
        readings_by_type = {}
        for reading in readings:
            # Convert to dict if it's a SensorReading object
            if hasattr(reading, 'to_dict'):
                reading = reading.to_dict()
                
            sensor_type = reading['sensor_type']
            if sensor_type not in readings_by_type:
                readings_by_type[sensor_type] = []
            readings_by_type[sensor_type].append(reading)
        
        # Process each sensor type
        results = []
        for sensor_type, type_readings in readings_by_type.items():
            # Convert to DataFrame
            df = pd.DataFrame(type_readings)
            
            # Prepare features for ML
            features = ['value']
            
            # Add time-based features
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            features.extend(['hour', 'day_of_week'])
            
            # One-hot encode location
            location_dummies = pd.get_dummies(df['location'], prefix='location')
            df = pd.concat([df, location_dummies], axis=1)
            features.extend(location_dummies.columns.tolist())
            
            # Extract feature data
            X = df[features]
            
            # Get detection results from multiple models
            model_results = []
            
            # Try location-specific model first
            locations = df['location'].unique()
            for location in locations:
                location_df = df[df['location'] == location]
                location_X = location_df[features]
                
                # Skip if not enough data
                if len(location_df) < 5:
                    continue
                    
                # Get scores from location-specific model
                location_model = self.get_model('isolation_forest', sensor_type, location)
                if location_model and location_model.is_trained:
                    result = location_model.detect_anomalies(location_X)
                    
                    # Store results
                    result_dict = {
                        'idx': location_df.index.tolist(),
                        'anomaly_score': result['anomaly_score'].tolist(),
                        'is_anomaly': result['is_anomaly'].tolist()
                    }
                    model_results.append(result_dict)
            
            # If we have location-specific results for all rows, no need to use general model
            if sum(len(r['idx']) for r in model_results) == len(df):
                # Merge all location-specific results
                merged_scores = np.zeros(len(df))
                merged_flags = np.zeros(len(df), dtype=bool)
                
                for result in model_results:
                    for i, idx in enumerate(result['idx']):
                        merged_scores[idx] = result['anomaly_score'][i]
                        merged_flags[idx] = result['is_anomaly'][i]
                
                # Add results to dataframe
                df['anomaly_score'] = merged_scores
                df['is_anomaly'] = merged_flags
            else:
                # Use general model as fallback
                general_model = self.get_model('isolation_forest', sensor_type)
                if general_model and general_model.is_trained:
                    result = general_model.detect_anomalies(X)
                    df['anomaly_score'] = result['anomaly_score']
                    df['is_anomaly'] = result['is_anomaly']
                else:
                    # No trained model available, use the flag from the data itself
                    df['anomaly_score'] = 0.0
                    df['is_anomaly'] = df.get('is_anomaly', False)
            
            # Convert back to list of dictionaries
            for _, row in df.iterrows():
                # Convert to dict and add to results
                result_dict = row.to_dict()
                results.append(result_dict)
        
        return results


# Example usage
if __name__ == "__main__":
    # Create the anomaly detector
    detector = AnomalyDetector()
    
    # Load some test data
    data = detector.load_sensor_data(window_seconds=3600)
    
    if data:
        # Train the models
        detector.train_models()
        
        # Wait for training to complete
        while detector.running:
            print("Waiting for training to complete...")
            time.sleep(1)
        
        # Generate some test readings
        from sensors.sensor_simulator import SensorManager
        manager = SensorManager()
        readings = manager.get_all_readings()
        
        # Detect anomalies
        results = detector.detect_anomalies([r.to_dict() for r in readings])
        
        # Print results
        for result in results:
            if result['is_anomaly']:
                print(f"Anomaly detected: {result['sensor_id']} with score {result['anomaly_score']:.4f}")
