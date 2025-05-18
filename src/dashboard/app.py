"""
Dashboard for Smart Home Monitoring System

This module implements a web-based dashboard using Dash and Plotly
for visualizing sensor data and anomalies.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import requests
import socketio

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_UPDATE_INTERVAL, 
    API_HOST, API_PORT, SENSOR_TYPES, SENSOR_LOCATIONS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')

# Initialize the Dash app
app = dash.Dash(
    __name__,
    title='Smart Home Monitoring System',
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Initialize SocketIO client for real-time updates
sio = socketio.Client()

# Base URL for API calls
API_BASE_URL = f'http://{API_HOST}:{API_PORT}'

# Define app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Smart Home Monitoring System', className='dashboard-title'),
        html.H2('AI-Powered Anomaly Detection', className='dashboard-subtitle'),
        
        # Control panel with filters
        html.Div([
            html.Div([
                html.Label('Sensor Type:'),
                dcc.Dropdown(
                    id='sensor-type-dropdown',
                    options=[{'label': t.capitalize(), 'value': t} for t in SENSOR_TYPES],
                    value='temperature',
                    clearable=False
                )
            ], className='filter-item'),
            
            html.Div([
                html.Label('Location:'),
                dcc.Dropdown(
                    id='location-dropdown',
                    options=[{'label': l.replace('_', ' ').capitalize(), 'value': l} for l in SENSOR_LOCATIONS],
                    value='all',
                    clearable=False
                )
            ], className='filter-item'),
            
            html.Div([
                html.Label('Time Window:'),
                dcc.Dropdown(
                    id='time-window-dropdown',
                    options=[
                        {'label': 'Last 15 Minutes', 'value': 15*60},
                        {'label': 'Last Hour', 'value': 60*60},
                        {'label': 'Last 6 Hours', 'value': 6*60*60},
                        {'label': 'Last 24 Hours', 'value': 24*60*60}
                    ],
                    value=60*60,
                    clearable=False
                )
            ], className='filter-item')
        ], className='filter-container')
    ], className='dashboard-header'),
    
    # Main content
    html.Div([
        # Sensor readings graph
        html.Div([
            html.Div([
                html.H3('Sensor Readings', className='panel-title'),
                dcc.Loading(
                    id="loading-readings-graph",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id='readings-graph',
                            config={'displayModeBar': False}
                        )
                    ]
                )
            ], className='panel-content')
        ], className='dashboard-panel'),
        
        # Anomaly detection graph
        html.Div([
            html.Div([
                html.H3('Anomaly Detection', className='panel-title'),
                dcc.Loading(
                    id="loading-anomaly-graph",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id='anomaly-graph',
                            config={'displayModeBar': False}
                        )
                    ]
                )
            ], className='panel-content')
        ], className='dashboard-panel')
    ], className='main-content'),
    
    # Bottom panels
    html.Div([
        # Stats panel
        html.Div([
            html.Div([
                html.H3('System Statistics', className='panel-title'),
                html.Div(id='stats-container', className='stats-grid')
            ], className='panel-content')
        ], className='dashboard-panel'),
        
        # Recent anomalies
        html.Div([
            html.Div([
                html.H3('Recent Anomalies', className='panel-title'),
                html.Div(id='anomalies-container', className='anomalies-list')
            ], className='panel-content')
        ], className='dashboard-panel')
    ], className='bottom-panels'),
    
    # Hidden components for storing state
    html.Div([
        dcc.Store(id='sensor-data-store'),
        dcc.Store(id='anomaly-data-store'),
        dcc.Store(id='stats-data-store'),
        dcc.Interval(
            id='refresh-interval',
            interval=DASHBOARD_UPDATE_INTERVAL * 1000,  # in milliseconds
            n_intervals=0
        )
    ], style={'display': 'none'})
], className='dashboard-container')

@app.callback(
    [
        Output('sensor-data-store', 'data'),
        Output('anomaly-data-store', 'data'),
        Output('stats-data-store', 'data')
    ],
    [
        Input('refresh-interval', 'n_intervals'),
        Input('sensor-type-dropdown', 'value'),
        Input('location-dropdown', 'value'),
        Input('time-window-dropdown', 'value')
    ]
)
def update_data_stores(n_intervals, sensor_type, location, time_window):
    """Update data stores with current sensor readings and anomalies."""
    # Skip the initial callback
    if n_intervals == 0:
        raise PreventUpdate
    
    # Calculate time window
    to_time = time.time()
    from_time = to_time - time_window
    
    # Prepare parameters
    params = {
        'from_time': from_time,
        'to_time': to_time,
        'limit': 1000
    }
    
    # Add location filter if not 'all'
    if location != 'all':
        params['location'] = location
    
    # Fetch sensor data
    sensor_data = []
    try:
        if sensor_type == 'all':
            # Fetch data for all sensor types
            for st in SENSOR_TYPES:
                response = requests.get(f'{API_BASE_URL}/api/sensors/{st}', params=params)
                if response.status_code == 200:
                    sensor_data.extend(response.json())
        else:
            # Fetch data for specific sensor type
            response = requests.get(f'{API_BASE_URL}/api/sensors/{sensor_type}', params=params)
            if response.status_code == 200:
                sensor_data = response.json()
    except Exception as e:
        logger.error(f"Error fetching sensor data: {e}")
    
    # Fetch anomaly data
    anomaly_data = []
    try:
        # Prepare anomaly parameters
        anomaly_params = params.copy()
        if sensor_type != 'all':
            anomaly_params['sensor_type'] = sensor_type
        
        response = requests.get(f'{API_BASE_URL}/api/anomalies', params=anomaly_params)
        if response.status_code == 200:
            anomaly_data = response.json()
    except Exception as e:
        logger.error(f"Error fetching anomaly data: {e}")
    
    # Fetch system stats
    stats_data = {}
    try:
        response = requests.get(f'{API_BASE_URL}/api/stats')
        if response.status_code == 200:
            stats_data = response.json()
    except Exception as e:
        logger.error(f"Error fetching system stats: {e}")
    
    return sensor_data, anomaly_data, stats_data

@app.callback(
    Output('readings-graph', 'figure'),
    [Input('sensor-data-store', 'data'),
     Input('sensor-type-dropdown', 'value'),
     Input('location-dropdown', 'value')]
)
def update_readings_graph(sensor_data, sensor_type, location):
    """Update the sensor readings graph."""
    if not sensor_data:
        return {
            'data': [],
            'layout': {
                'title': 'No sensor data available',
                'showlegend': False
            }
        }
    
    # Create a DataFrame from the sensor data
    df = pd.DataFrame(sensor_data)
    
    # Filter by location if not 'all'
    if location != 'all':
        df = df[df['location'] == location]
    
    # Skip if no data after filtering
    if df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No data for the selected filters',
                'showlegend': False
            }
        }
    
    # Convert timestamps to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Group by location and sensor_id
    traces = []
    
    # For each location and sensor id, create a trace
    for (loc, sensor_id), group in df.groupby(['location', 'sensor_id']):
        # Sort by datetime
        group = group.sort_values('datetime')
        
        sensor_type = group['sensor_type'].iloc[0]
        
        traces.append(
            go.Scatter(
                x=group['datetime'],
                y=group['value'],
                mode='lines+markers',
                name=f'{loc.replace("_", " ").capitalize()} ({sensor_type.capitalize()})',
                line=dict(width=2),
                marker=dict(
                    size=6,
                    symbol='circle'
                )
            )
        )
    
    # Create a title based on sensor type
    if sensor_type == 'all':
        title = 'All Sensor Readings'
    else:
        title = f'{sensor_type.capitalize()} Sensor Readings'
    
    # Add location to title if specified
    if location != 'all':
        title += f' in {location.replace("_", " ").capitalize()}'
    
    # Create the graph layout
    layout = {
        'title': title,
        'xaxis': {
            'title': 'Time',
            'gridcolor': 'rgba(230, 230, 230, 0.5)'
        },
        'yaxis': {
            'title': get_unit_label(sensor_type),
            'gridcolor': 'rgba(230, 230, 230, 0.5)'
        },
        'margin': {'l': 40, 'b': 40, 't': 50, 'r': 10},
        'hovermode': 'closest',
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0)',
        'showlegend': True,
        'legend': {'orientation': 'h', 'y': -0.2}
    }
    
    return {'data': traces, 'layout': layout}

@app.callback(
    Output('anomaly-graph', 'figure'),
    [Input('sensor-data-store', 'data'),
     Input('anomaly-data-store', 'data'),
     Input('sensor-type-dropdown', 'value'),
     Input('location-dropdown', 'value')]
)
def update_anomaly_graph(sensor_data, anomaly_data, sensor_type, location):
    """Update the anomaly detection graph."""
    if not sensor_data or not anomaly_data:
        return {
            'data': [],
            'layout': {
                'title': 'No anomaly data available',
                'showlegend': False
            }
        }
    
    # Create DataFrames from the data
    sensor_df = pd.DataFrame(sensor_data)
    anomaly_df = pd.DataFrame(anomaly_data)
    
    # Filter by location if not 'all'
    if location != 'all':
        sensor_df = sensor_df[sensor_df['location'] == location]
        anomaly_df = anomaly_df[anomaly_df['location'] == location]
    
    # Skip if no data after filtering
    if sensor_df.empty and anomaly_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No anomaly data for the selected filters',
                'showlegend': False
            }
        }
    
    # Convert timestamps to datetime
    sensor_df['datetime'] = pd.to_datetime(sensor_df['datetime'])
    anomaly_df['datetime'] = pd.to_datetime(anomaly_df['datetime'])
    
    # Create traces
    traces = []
    
    # Add regular sensor readings
    for (loc, sensor_id), group in sensor_df.groupby(['location', 'sensor_id']):
        # Skip if no data
        if group.empty:
            continue
            
        # Sort by datetime
        group = group.sort_values('datetime')
        
        sensor_type = group['sensor_type'].iloc[0]
        
        # Regular readings
        traces.append(
            go.Scatter(
                x=group['datetime'],
                y=group['value'],
                mode='lines',
                name=f'{loc.replace("_", " ").capitalize()} (Normal)',
                line=dict(width=2, color='rgba(0, 128, 0, 0.5)'),
                hoverinfo='skip'
            )
        )
    
    # Add anomalies
    for (loc, sensor_id), group in anomaly_df.groupby(['location', 'sensor_id']):
        # Skip if no data
        if group.empty:
            continue
            
        # Sort by datetime
        group = group.sort_values('datetime')
        
        sensor_type = group['sensor_type'].iloc[0]
        
        # Anomalies
        traces.append(
            go.Scatter(
                x=group['datetime'],
                y=group['value'],
                mode='markers',
                name=f'{loc.replace("_", " ").capitalize()} (Anomaly)',
                marker=dict(
                    size=10,
                    symbol='x',
                    color='rgba(255, 0, 0, 0.8)',
                    line=dict(width=2, color='rgba(255, 0, 0, 1)')
                ),
                hovertext=[f"Anomaly Score: {score:.4f}" for score in group['anomaly_score']],
                hoverinfo='text+x+y'
            )
        )
    
    # Create a title based on sensor type
    if sensor_type == 'all':
        title = 'Anomaly Detection - All Sensors'
    else:
        title = f'Anomaly Detection - {sensor_type.capitalize()} Sensors'
    
    # Add location to title if specified
    if location != 'all':
        title += f' in {location.replace("_", " ").capitalize()}'
    
    # Create the graph layout
    layout = {
        'title': title,
        'xaxis': {
            'title': 'Time',
            'gridcolor': 'rgba(230, 230, 230, 0.5)'
        },
        'yaxis': {
            'title': get_unit_label(sensor_type),
            'gridcolor': 'rgba(230, 230, 230, 0.5)'
        },
        'margin': {'l': 40, 'b': 40, 't': 50, 'r': 10},
        'hovermode': 'closest',
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0)',
        'showlegend': True,
        'legend': {'orientation': 'h', 'y': -0.2}
    }
    
    return {'data': traces, 'layout': layout}

@app.callback(
    Output('stats-container', 'children'),
    [Input('stats-data-store', 'data')]
)
def update_stats_container(stats_data):
    """Update the system statistics container."""
    if not stats_data:
        return html.Div('No statistics available')
    
    stats_items = []
    
    # Total sensors
    stats_items.append(
        html.Div([
            html.H4('Total Sensors'),
            html.P(str(stats_data.get('sensors', {}).get('total', 0)))
        ], className='stat-card')
    )
    
    # Total readings
    stats_items.append(
        html.Div([
            html.H4('Total Readings'),
            html.P(str(stats_data.get('readings', {}).get('total', 0)))
        ], className='stat-card')
    )
    
    # Total anomalies
    stats_items.append(
        html.Div([
            html.H4('Total Anomalies'),
            html.P(str(stats_data.get('anomalies', {}).get('total', 0)))
        ], className='stat-card')
    )
    
    # Recent anomalies (last hour)
    stats_items.append(
        html.Div([
            html.H4('Recent Anomalies'),
            html.P(str(stats_data.get('anomalies', {}).get('recent_count', 0)))
        ], className='stat-card')
    )
    
    # System uptime
    uptime_seconds = stats_data.get('system', {}).get('uptime', 0)
    uptime_str = format_uptime(uptime_seconds)
    stats_items.append(
        html.Div([
            html.H4('System Uptime'),
            html.P(uptime_str)
        ], className='stat-card')
    )
    
    return html.Div(stats_items, className='stats-grid')

@app.callback(
    Output('anomalies-container', 'children'),
    [Input('anomaly-data-store', 'data')]
)
def update_anomalies_container(anomaly_data):
    """Update the recent anomalies container."""
    if not anomaly_data:
        return html.Div('No anomalies detected')
    
    # Sort by timestamp descending
    anomaly_data = sorted(anomaly_data, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    # Limit to 10 most recent anomalies
    anomaly_data = anomaly_data[:10]
    
    anomaly_items = []
    
    for anomaly in anomaly_data:
        # Format the anomaly data
        sensor_type = anomaly.get('sensor_type', '').capitalize()
        location = anomaly.get('location', '').replace('_', ' ').capitalize()
        value = anomaly.get('value', 0)
        anomaly_score = anomaly.get('anomaly_score', 0)
        timestamp = datetime.fromtimestamp(anomaly.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
        
        anomaly_items.append(
            html.Div([
                html.Div([
                    html.H4(f"{sensor_type} in {location}"),
                    html.P(f"Value: {value} {get_unit(sensor_type)}")
                ]),
                html.Div([
                    html.P(f"Score: {anomaly_score:.4f}"),
                    html.P(f"Time: {timestamp}")
                ])
            ], className='anomaly-card')
        )
    
    return html.Div(anomaly_items, className='anomalies-list')

def get_unit(sensor_type):
    """Get the unit for a sensor type."""
    units = {
        'temperature': '°C',
        'humidity': '%',
        'co2': 'ppm',
        'smoke': 'ppm',
        'light': 'lux',
        'motion': '',
        'door': '',
        'window': ''
    }
    return units.get(sensor_type.lower(), '')

def get_unit_label(sensor_type):
    """Get the unit label for a sensor type."""
    labels = {
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'co2': 'CO2 Level (ppm)',
        'smoke': 'Smoke Level (ppm)',
        'light': 'Light Level (lux)',
        'motion': 'Motion Detected',
        'door': 'Door Status (0=Closed, 1=Open)',
        'window': 'Window Status (0=Closed, 1=Open)'
    }
    return labels.get(sensor_type.lower(), 'Value')

def format_uptime(seconds):
    """Format uptime in seconds to a human-readable string."""
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

@sio.on('connect')
def on_connect():
    """Handle socket.io connection."""
    logger.info("Connected to API server")

@sio.on('disconnect')
def on_disconnect():
    """Handle socket.io disconnection."""
    logger.info("Disconnected from API server")

@sio.on('readings')
def on_readings(data):
    """Handle new readings event."""
    logger.debug(f"Received {len(data)} new readings")

@sio.on('anomaly')
def on_anomaly(data):
    """Handle new anomaly event."""
    logger.info(f"Received new anomaly: {data}")

def start_dashboard(host=None, port=None):
    """
    Start the dashboard server.
    
    Args:
        host (str, optional): Host to bind to
        port (int, optional): Port to bind to
    """
    host = host or DASHBOARD_HOST
    port = port or DASHBOARD_PORT
    
    # Try to connect to the API server via Socket.IO
    try:
        sio.connect(f'http://{API_HOST}:{API_PORT}')
    except Exception as e:
        logger.warning(f"Could not connect to API server: {e}")
    
    # Add some CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .dashboard-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .dashboard-header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .dashboard-title {
                    margin: 0;
                    font-size: 24px;
                }
                .dashboard-subtitle {
                    margin: 5px 0 20px;
                    font-size: 16px;
                    font-weight: normal;
                    opacity: 0.8;
                }
                .filter-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-top: 15px;
                }
                .filter-item {
                    flex: 1;
                    min-width: 200px;
                }
                .dashboard-panel {
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    overflow: hidden;
                }
                .panel-content {
                    padding: 15px;
                }
                .panel-title {
                    margin-top: 0;
                    margin-bottom: 15px;
                    font-size: 18px;
                    color: #2c3e50;
                }
                .main-content {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
                .bottom-panels {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                    gap: 15px;
                }
                .stat-card {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 10px;
                    text-align: center;
                }
                .stat-card h4 {
                    margin: 0 0 5px;
                    font-size: 14px;
                    color: #6c757d;
                }
                .stat-card p {
                    margin: 0;
                    font-size: 20px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .anomalies-list {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    max-height: 400px;
                    overflow-y: auto;
                }
                .anomaly-card {
                    background-color: #f8d7da;
                    border-radius: 8px;
                    padding: 10px;
                    display: flex;
                    justify-content: space-between;
                }
                .anomaly-card h4 {
                    margin: 0 0 5px;
                    font-size: 14px;
                    color: #721c24;
                }
                .anomaly-card p {
                    margin: 0;
                    font-size: 12px;
                    color: #721c24;
                }
                @media (max-width: 768px) {
                    .bottom-panels {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Start the dashboard server
    logger.info(f"Starting dashboard on {host}:{port}")
    app.run_server(host=host, port=port, debug=False)

# For testing
if __name__ == '__main__':
    start_dashboard()
