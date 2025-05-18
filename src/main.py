"""
Main entry point for the Smart Home Monitoring System

This module initializes and starts all components of the system.
"""
import os
import sys
import time
import logging
import threading
import argparse
from dotenv import load_dotenv

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file if it exists
load_dotenv()

from config import API_HOST, API_PORT, DASHBOARD_HOST, DASHBOARD_PORT
from sensors.sensor_simulator import SensorManager
from ml.anomaly_detector import AnomalyDetector
from api.routes import start_api_server, stop_api_server
from dashboard.app import start_dashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Smart Home Monitoring System')
    
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Do not start the dashboard')
    
    parser.add_argument('--api-host', type=str, default=API_HOST,
                        help=f'Host for the API server (default: {API_HOST})')
    
    parser.add_argument('--api-port', type=int, default=API_PORT,
                        help=f'Port for the API server (default: {API_PORT})')
    
    parser.add_argument('--dashboard-host', type=str, default=DASHBOARD_HOST,
                        help=f'Host for the dashboard (default: {DASHBOARD_HOST})')
    
    parser.add_argument('--dashboard-port', type=int, default=DASHBOARD_PORT,
                        help=f'Port for the dashboard (default: {DASHBOARD_PORT})')
    
    parser.add_argument('--train', action='store_true',
                        help='Train the anomaly detection models before starting')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting Smart Home Monitoring System")
    
    # Initialize the anomaly detector
    anomaly_detector = AnomalyDetector()
    
    # Train models if requested
    if args.train:
        logger.info("Training anomaly detection models")
        anomaly_detector.train_models()
        
        while anomaly_detector.running:
            logger.info("Waiting for training to complete...")
            time.sleep(1)
    
    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=start_api_server,
        args=(args.api_host, args.api_port),
        daemon=True
    )
    api_thread.start()
    logger.info(f"API server started on {args.api_host}:{args.api_port}")
    
    # Start dashboard if not disabled
    if not args.no_dashboard:
        dashboard_thread = threading.Thread(
            target=start_dashboard,
            args=(args.dashboard_host, args.dashboard_port),
            daemon=True
        )
        dashboard_thread.start()
        logger.info(f"Dashboard started on {args.dashboard_host}:{args.dashboard_port}")
        
        # Open dashboard in web browser (optional)
        import webbrowser
        webbrowser.open(f"http://{args.dashboard_host}:{args.dashboard_port}")
    
    # Print usage instructions
    print("\n" + "="*80)
    print("Smart Home Monitoring System is running!")
    print("="*80)
    print(f"API Server: http://{args.api_host}:{args.api_port}")
    
    if not args.no_dashboard:
        print(f"Dashboard: http://{args.dashboard_host}:{args.dashboard_port}")
    
    print("\nPress Ctrl+C to stop the system.")
    print("="*80 + "\n")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping Smart Home Monitoring System")
        stop_api_server()
        
    logger.info("System stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
