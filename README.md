# Smart Home Monitoring System with AI-Powered Anomaly Detection

An advanced IoT project that demonstrates integrating multiple IoT sensors with AI for anomaly detection.

## Features

- **Multi-sensor Data Collection**: Simulates various home IoT sensors (temperature, humidity, motion, etc.)
- **Real-time Data Processing**: Processes sensor data streams using modern data pipeline techniques
- **AI-Powered Anomaly Detection**: Uses machine learning to detect unusual patterns in sensor data
- **Interactive Dashboard**: Visualizes sensor data and anomaly alerts in real-time
- **RESTful API**: Provides programmatic access to sensor data and anomaly detection results
- **Edge Computing Simulation**: Demonstrates edge computing principles with local preprocessing

## Architecture

```
                   ┌─────────────┐
                   │IoT Sensors  │
                   │(Simulated)  │
                   └──────┬──────┘
                          │
                          ▼
┌────────────┐    ┌──────────────┐    ┌────────────┐
│ Dashboard  │◄───┤ Data Pipeline ├───►│ AI Models  │
└────────────┘    └──────┬───────┘    └────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ REST API    │
                  └─────────────┘
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/smart-home-monitor.git
cd smart-home-monitor
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python src/main.py
```

4. Open the dashboard in your browser:
```
http://localhost:8050
```

## Components

1. **Sensor Simulator**: Generates realistic IoT sensor data
2. **Anomaly Detector**: Uses machine learning to identify unusual patterns
3. **Web Dashboard**: Interactive visualization of sensor data and anomalies
4. **REST API**: Interface for external applications

## Usage Example

```python
# Use the API to get sensor data
import requests

# Get the latest temperature readings
response = requests.get('http://localhost:5000/api/sensors/temperature')
temperature_data = response.json()
print(temperature_data)
```

## Google Interview Relevance

This project demonstrates proficiency in:
- Python programming with modern best practices
- Machine Learning and AI implementation
- IoT systems architecture
- Real-time data processing
- Full-stack application development
- Edge computing concepts
