#!/usr/bin/env python3
"""
Main execution script for comprehensive flight black box anomaly detection
Combines all algorithms and provides a unified interface
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import main detector
from flight_anomaly_detector import FlightAnomalyDetector

# Import advanced methods if available
try:
    from advanced_anomaly_methods import integrate_advanced_methods
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("Advanced methods not available. Using basic methods only.")

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'results', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Directories setup completed.")

def find_csv_files(data_dir='data'):
    """Find all CSV files in the data directory"""
    csv_files = []
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return csv_files

def validate_csv_file(csv_path):
    """Validate the CSV file format for flight data"""
    try:
        # Read first few rows to check format
        df = pd.read_csv(csv_path, nrows=5)
        
        print(f"CSV Validation for {csv_path}:")
        print(f"  - Shape (first 5 rows): {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Data types: {df.dtypes.to_dict()}")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print("  - WARNING: No numeric columns found!")
            return False
        else:
            print(f"  - Numeric columns: {len(numeric_cols)}")
        
        # Check for time columns
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower() or 'date' in col.lower()]
        if time_cols:
            print(f"  - Time columns detected: {time_cols}")
        else:
            print("  - No time columns detected (using index as time)")
        
        return True
        
    except Exception as e:
        print(f"CSV validation failed: {e}")
        return False

def create_sample_data():
    """Create sample flight data for testing"""
    print("Creating sample flight black box data for testing...")
    
    # Generate synthetic flight data
    n_samples = 10000
    time_index = pd.date_range('2023-01-01 10:00:00', periods=n_samples, freq='1S')
    
    # Base flight parameters
    np.random.seed(42)
    
    # Normal flight phase (first 80%)
    normal_samples = int(0.8 * n_samples)
    
    # Flight parameters during normal operation
    altitude = 35000 + np.random.normal(0, 100, normal_samples)
    airspeed = 450 + np.random.normal(0, 20, normal_samples)
    heading = 270 + np.cumsum(np.random.normal(0, 0.5, normal_samples))
    pitch = np.random.normal(2, 1, normal_samples)
    roll = np.random.normal(0, 2, normal_samples)
    
    # Engine parameters
    engine1_temp = 800 + np.random.normal(0, 30, normal_samples)
    engine2_temp = 805 + np.random.normal(0, 30, normal_samples)
    fuel_flow = 2500 + np.random.normal(0, 100, normal_samples)
    
    # Control surface positions
    elevator = np.random.normal(0, 5, normal_samples)
    rudder = np.random.normal(0, 3, normal_samples)
    aileron = np.random.normal(0, 4, normal_samples)
    
    # Abnormal flight phase (last 20%) - introduce anomalies
    abnormal_samples = n_samples - normal_samples
    
    # Sudden altitude loss
    altitude_abn = altitude[-1] + np.cumsum(np.random.normal(-50, 50, abnormal_samples))
    
    # Erratic airspeed
    airspeed_abn = airspeed[-1] + np.cumsum(np.random.normal(0, 30, abnormal_samples))
    
    # Sharp heading changes
    heading_abn = heading[-1] + np.cumsum(np.random.normal(0, 5, abnormal_samples))
    
    # Unusual pitch and roll
    pitch_abn = np.random.normal(15, 8, abnormal_samples)  # Steep climb/dive
    roll_abn = np.random.normal(0, 15, abnormal_samples)   # Banking
    
    # Engine problems
    engine1_temp_abn = engine1_temp[-1] + np.cumsum(np.random.normal(20, 40, abnormal_samples))
    engine2_temp_abn = engine2_temp[-1] + np.cumsum(np.random.normal(5, 20, abnormal_samples))
    fuel_flow_abn = fuel_flow[-1] + np.cumsum(np.random.normal(-10, 50, abnormal_samples))
    
    # Control surface deflections
    elevator_abn = np.random.normal(0, 15, abnormal_samples)
    rudder_abn = np.random.normal(0, 10, abnormal_samples)
    aileron_abn = np.random.normal(0, 12, abnormal_samples)
    
    # Combine normal and abnormal data
    flight_data = pd.DataFrame({
        'timestamp': time_index,
        'altitude': np.concatenate([altitude, altitude_abn]),
        'airspeed': np.concatenate([airspeed, airspeed_abn]),
        'heading': np.concatenate([heading, heading_abn]),
        'pitch': np.concatenate([pitch, pitch_abn]),
        'roll': np.concatenate([roll, roll_abn]),
        'engine1_temperature': np.concatenate([engine1_temp, engine1_temp_abn]),
        'engine2_temperature': np.concatenate([engine2_temp, engine2_temp_abn]),
        'fuel_flow_rate': np.concatenate([fuel_flow, fuel_flow_abn]),
        'elevator_position': np.concatenate([elevator, elevator_abn]),
        'rudder_position': np.concatenate([rudder, rudder_abn]),
        'aileron_position': np.concatenate([aileron, aileron_abn]),
        'is_anomaly': np.concatenate([np.zeros(normal_samples), np.ones(abnormal_samples)])
    })
    
    # Save to CSV
    sample_path = 'data/sample_flight_data.csv'
    flight_data.to_csv(sample_path, index=False)
    
    print(f"Sample flight data created: {sample_path}")
    print(f"Data contains {abnormal_samples} anomalous samples out of {n_samples} total")
    
    return sample_path

def run_comprehensive_analysis(csv_path, use_advanced=True):
    """Run comprehensive anomaly detection analysis"""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE FLIGHT BLACK BOX ANOMALY DETECTION")
    print(f"{'='*60}")
    
    # Validate CSV file
    if not validate_csv_file(csv_path):
        print("CSV validation failed. Cannot proceed.")
        return False
    
    # Initialize main detector
    print(f"\nInitializing detector with file: {csv_path}")
    detector = FlightAnomalyDetector(csv_path)
    
    # Load data
    if not detector.load_data():
        print("Failed to load data. Exiting.")
        return False
    
    print(f"Data loaded successfully:")
    print(f"  - Shape: {detector.data.shape}")
    print(f"  - Time range: {len(detector.data)} samples")
    print(f"  - Features: {list(detector.data.columns)}")
    
    # Run basic methods
    print(f"\n{'-'*40}")
    print("RUNNING BASIC ANOMALY DETECTION METHODS")
    print(f"{'-'*40}")
    
    detector.run_all_methods()
    
    # Run advanced methods if available and requested
    if use_advanced and ADVANCED_AVAILABLE:
        print(f"\n{'-'*40}")
        print("RUNNING ADVANCED ANOMALY DETECTION METHODS")
        print(f"{'-'*40}")
        
        try:
            advanced_results = integrate_advanced_methods(detector, csv_path)
            
            # Update ensemble with advanced methods
            detector.ensemble_detection()
            
            print("Advanced methods completed successfully!")
            
        except Exception as e:
            print(f"Advanced methods failed: {e}")
            print("Continuing with basic methods only.")
    
    # Final ensemble and visualization
    print(f"\n{'-'*40}")
    print("GENERATING FINAL RESULTS")
    print(f"{'-'*40}")
    
    # Create final ensemble
    detector.ensemble_detection()
    
    # Generate comprehensive report
    report = detector.generate_report()
    
    # Print summary
    print(f"\nANOMALY DETECTION SUMMARY:")
    print(f"{'Method':<25} {'Anomalies':<10} {'Percentage':<12}")
    print(f"{'-'*50}")
    
    for method, results in detector.results.items():
        if 'anomaly_score' in results:
            n_anomalies = results['n_anomalies']
            percentage = (n_anomalies / len(detector.data)) * 100
            print(f"{method:<25} {n_anomalies:<10} {percentage:<12.2f}%")
    
    print(f"\nResults saved to:")
    print(f"  - Detailed report: results/anomaly_detection_report.txt")
    print(f"  - Anomaly scores: results/detailed_anomaly_scores.csv")
    print(f"  - Visualization: results/anomaly_detection_results.png")
    
    return True

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Flight Black Box Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_anomaly_detection.py                    # Use first CSV found in data/
  python run_anomaly_detection.py --file flight.csv  # Use specific file
  python run_anomaly_detection.py --create-sample    # Create sample data
  python run_anomaly_detection.py --no-advanced      # Skip advanced methods
        """
    )
    
    parser.add_argument('--file', '-f', 
                       help='Specific CSV file to analyze (in data/ directory)')
    parser.add_argument('--create-sample', '-s', action='store_true',
                       help='Create sample flight data for testing')
    parser.add_argument('--no-advanced', action='store_true',
                       help='Skip advanced detection methods')
    parser.add_argument('--list-files', '-l', action='store_true',
                       help='List available CSV files in data/ directory')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # List files if requested
    if args.list_files:
        csv_files = find_csv_files()
        if csv_files:
            print("Available CSV files in data/ directory:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file}")
        else:
            print("No CSV files found in data/ directory.")
        return
    
    # Create sample data if requested
    if args.create_sample:
        sample_path = create_sample_data()
        print(f"\nSample data created. You can now run:")
        print(f"python run_anomaly_detection.py --file {os.path.basename(sample_path)}")
        return
    
    # Find CSV file to use
    csv_path = None
    
    if args.file:
        # Use specified file
        csv_path = os.path.join('data', args.file)
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            return
    else:
        # Find first available CSV file
        csv_files = find_csv_files()
        if not csv_files:
            print("No CSV files found in data/ directory!")
            print("Options:")
            print("  1. Place your flight data CSV in the data/ directory")
            print("  2. Use --create-sample to generate test data")
            print("  3. Use --help for more options")
            return
        
        csv_path = os.path.join('data', csv_files[0])
        print(f"Using CSV file: {csv_path}")
    
    # Run comprehensive analysis
    use_advanced = not args.no_advanced
    success = run_comprehensive_analysis(csv_path, use_advanced=use_advanced)
    
    if success:
        print(f"\n{'='*60}")
        print("ANOMALY DETECTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("1. Review the generated report in results/")
        print("2. Examine the anomaly scores CSV for detailed analysis")
        print("3. Check the visualization plots")
        print("4. Investigate high-scoring time periods in your data")
    else:
        print("Anomaly detection failed. Please check your data and try again.")


if __name__ == "__main__":
    main()