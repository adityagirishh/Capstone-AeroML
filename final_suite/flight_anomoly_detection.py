#!/usr/bin/env python3
"""
Comprehensive Flight Black Box Anomaly Detection System
Combines multiple algorithms for robust anomaly detection in time series data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some algorithms will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Some algorithms will be disabled.")

class FlightAnomalyDetector:
    """Main class for flight black box anomaly detection"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the flight data"""
        print("Loading flight black box data...")
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Data shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Handle time column if present
            time_cols = [col for col in self.data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
            if time_cols:
                self.data[time_cols[0]] = pd.to_datetime(self.data[time_cols[0]], errors='coerce')
                self.data = self.data.set_index(time_cols[0])
            
            # Select numeric columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data = self.data[numeric_cols]
            
            # Handle missing values
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            # Scale the data
            self.scaled_data = self.scaler.fit_transform(self.data)
            
            print("Data loaded and preprocessed successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def dbscan_anomaly_detection(self, eps=0.5, min_samples=5):
        """DBSCAN clustering for anomaly detection"""
        print("Running DBSCAN anomaly detection...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.scaled_data)
        
        # Points with label -1 are outliers/anomalies
        anomalies = labels == -1
        anomaly_score = np.where(anomalies, 1, 0)
        
        self.results['DBSCAN'] = {
            'anomaly_score': anomaly_score,
            'n_anomalies': np.sum(anomalies),
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }
        
        print(f"DBSCAN: Found {np.sum(anomalies)} anomalies in {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return anomaly_score
    
    def kmeans_anomaly_detection(self, n_clusters=8, threshold_percentile=95):
        """K-means clustering with distance-based anomaly detection"""
        print("Running K-means anomaly detection...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.scaled_data)
        
        # Calculate distances to cluster centers
        distances = np.min(np.linalg.norm(self.scaled_data[:, np.newaxis] - kmeans.cluster_centers_, axis=2), axis=1)
        
        # Define threshold based on percentile
        threshold = np.percentile(distances, threshold_percentile)
        anomalies = distances > threshold
        anomaly_score = distances / np.max(distances)  # Normalize scores
        
        self.results['KMeans'] = {
            'anomaly_score': anomaly_score,
            'n_anomalies': np.sum(anomalies),
            'threshold': threshold
        }
        
        print(f"K-means: Found {np.sum(anomalies)} anomalies with threshold {threshold:.4f}")
        return anomaly_score
    
    def isolation_forest_detection(self, contamination=0.1):
        """Isolation Forest anomaly detection"""
        print("Running Isolation Forest anomaly detection...")
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(self.scaled_data)
        anomaly_scores = iso_forest.decision_function(self.scaled_data)
        
        # Convert predictions to binary (1 for anomaly, 0 for normal)
        anomalies = predictions == -1
        # Normalize scores to [0, 1]
        normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
        
        self.results['IsolationForest'] = {
            'anomaly_score': 1 - normalized_scores,  # Invert so higher score = more anomalous
            'n_anomalies': np.sum(anomalies),
            'raw_scores': anomaly_scores
        }
        
        print(f"Isolation Forest: Found {np.sum(anomalies)} anomalies")
        return 1 - normalized_scores
    
    def pca_reconstruction_detection(self, n_components=None, threshold_percentile=95):
        """PCA-based reconstruction anomaly detection"""
        print("Running PCA reconstruction anomaly detection...")
        
        if n_components is None:
            # Use 95% of variance
            pca_temp = PCA()
            pca_temp.fit(self.scaled_data)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.95) + 1
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(self.scaled_data)
        reconstructed = pca.inverse_transform(transformed)
        
        # Calculate reconstruction error
        reconstruction_error = np.sum((self.scaled_data - reconstructed) ** 2, axis=1)
        
        # Define threshold
        threshold = np.percentile(reconstruction_error, threshold_percentile)
        anomalies = reconstruction_error > threshold
        
        # Normalize scores
        normalized_scores = reconstruction_error / np.max(reconstruction_error)
        
        self.results['PCA_Reconstruction'] = {
            'anomaly_score': normalized_scores,
            'n_anomalies': np.sum(anomalies),
            'threshold': threshold,
            'n_components': n_components
        }
        
        print(f"PCA Reconstruction: Found {np.sum(anomalies)} anomalies using {n_components} components")
        return normalized_scores
    
    def lstm_autoencoder_detection(self, sequence_length=50, epochs=50):
        """LSTM Autoencoder anomaly detection"""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping LSTM Autoencoder.")
            return np.zeros(len(self.data))
            
        print("Running LSTM Autoencoder anomaly detection...")
        
        # Prepare sequences
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(len(data) - seq_length + 1):
                sequences.append(data[i:i + seq_length])
            return np.array(sequences)
        
        sequences = create_sequences(self.scaled_data, sequence_length)
        
        # Build LSTM Autoencoder
        input_dim = self.scaled_data.shape[1]
        
        # Encoder
        inputs = Input(shape=(sequence_length, input_dim))
        encoded = LSTM(64, activation='relu')(inputs)
        
        # Decoder
        decoded = RepeatVector(sequence_length)(encoded)
        decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(input_dim))(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train
        print("Training LSTM Autoencoder...")
        autoencoder.fit(sequences, sequences, epochs=epochs, batch_size=32, verbose=0)
        
        # Predict and calculate reconstruction error
        reconstructed = autoencoder.predict(sequences, verbose=0)
        reconstruction_error = np.mean(np.abs(sequences - reconstructed), axis=(1, 2))
        
        # Pad with zeros for the first sequence_length-1 points
        full_scores = np.zeros(len(self.data))
        full_scores[sequence_length-1:] = reconstruction_error
        
        # Normalize scores
        normalized_scores = full_scores / (np.max(full_scores) if np.max(full_scores) > 0 else 1)
        
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold
        
        self.results['LSTM_Autoencoder'] = {
            'anomaly_score': normalized_scores,
            'n_anomalies': np.sum(anomalies),
            'threshold': threshold
        }
        
        print(f"LSTM Autoencoder: Found {np.sum(anomalies)} anomalies")
        return normalized_scores
    
    def transformer_anomaly_detection(self, window_size=100):
        """Simple Transformer-based anomaly detection"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Skipping Transformer detection.")
            return np.zeros(len(self.data))
            
        print("Running Transformer anomaly detection...")
        
        # Simple implementation - using statistical measures as proxy
        # In practice, you would implement or use TranAD here
        
        # Create rolling windows and calculate statistical features
        window_scores = []
        for i in range(len(self.scaled_data) - window_size + 1):
            window = self.scaled_data[i:i + window_size]
            
            # Calculate various statistical measures
            mean_change = np.mean(np.abs(np.diff(window, axis=0)))
            std_change = np.std(np.diff(window, axis=0))
            correlation_change = np.mean(np.abs(np.corrcoef(window.T)))
            
            # Combined anomaly score
            score = mean_change + std_change + (1 - correlation_change if not np.isnan(correlation_change) else 0)
            window_scores.append(score)
        
        # Pad scores
        full_scores = np.zeros(len(self.data))
        full_scores[window_size//2:-window_size//2+1] = window_scores
        
        # Normalize
        normalized_scores = full_scores / (np.max(full_scores) if np.max(full_scores) > 0 else 1)
        
        threshold = np.percentile([s for s in window_scores if s > 0], 95)
        anomalies = np.array(window_scores) > threshold
        
        self.results['Transformer_Simple'] = {
            'anomaly_score': normalized_scores,
            'n_anomalies': np.sum(anomalies),
            'threshold': threshold
        }
        
        print(f"Transformer (Simple): Found {np.sum(anomalies)} anomalies")
        return normalized_scores
    
    def ensemble_detection(self):
        """Combine all detection methods using ensemble approach"""
        print("Creating ensemble anomaly detection...")
        
        # Collect all anomaly scores
        all_scores = []
        for method, results in self.results.items():
            if 'anomaly_score' in results:
                all_scores.append(results['anomaly_score'])
        
        if not all_scores:
            print("No detection methods have been run yet!")
            return np.zeros(len(self.data))
        
        # Convert to array and handle different lengths
        max_len = max(len(scores) for scores in all_scores)
        padded_scores = []
        
        for scores in all_scores:
            if len(scores) < max_len:
                padded = np.zeros(max_len)
                padded[:len(scores)] = scores
                padded_scores.append(padded)
            else:
                padded_scores.append(scores)
        
        scores_array = np.array(padded_scores)
        
        # Ensemble methods
        ensemble_mean = np.mean(scores_array, axis=0)
        ensemble_max = np.max(scores_array, axis=0)
        ensemble_median = np.median(scores_array, axis=0)
        
        # Voting-based ensemble
        threshold = 0.5
        binary_predictions = scores_array > threshold
        ensemble_vote = np.sum(binary_predictions, axis=0) / len(all_scores)
        
        self.results['Ensemble_Mean'] = {
            'anomaly_score': ensemble_mean,
            'n_anomalies': np.sum(ensemble_mean > threshold)
        }
        
        self.results['Ensemble_Vote'] = {
            'anomaly_score': ensemble_vote,
            'n_anomalies': np.sum(ensemble_vote > 0.5)
        }
        
        print(f"Ensemble Mean: Found {np.sum(ensemble_mean > threshold)} anomalies")
        print(f"Ensemble Vote: Found {np.sum(ensemble_vote > 0.5)} anomalies")
        
        return ensemble_mean, ensemble_vote
    
    def visualize_results(self, save_plots=True):
        """Visualize anomaly detection results"""
        print("Creating visualization plots...")
        
        n_methods = len(self.results)
        if n_methods == 0:
            print("No results to visualize!")
            return
        
        fig, axes = plt.subplots(n_methods, 1, figsize=(15, 4 * n_methods))
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, results) in enumerate(self.results.items()):
            if 'anomaly_score' in results:
                scores = results['anomaly_score']
                
                # Plot time series with anomaly scores
                ax1 = axes[i]
                
                # Plot first few columns of original data
                for j, col in enumerate(self.data.columns[:min(3, len(self.data.columns))]):
                    ax1.plot(self.data.index if hasattr(self.data, 'index') else range(len(self.data)), 
                            self.data[col], alpha=0.6, label=col)
                
                # Highlight anomalies
                threshold = 0.5
                anomaly_indices = np.where(scores > threshold)[0]
                if len(anomaly_indices) > 0:
                    ax1.scatter(anomaly_indices, 
                               [self.data.iloc[idx, 0] for idx in anomaly_indices], 
                               c='red', s=50, alpha=0.8, label='Anomalies')
                
                ax1.set_title(f'{method} - Anomaly Detection ({results["n_anomalies"]} anomalies)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('results/anomaly_detection_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved to results/anomaly_detection_results.png")
        
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive anomaly detection report"""
        print("Generating comprehensive report...")
        
        report = {
            'dataset_info': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'time_range': f"{self.data.index[0]} to {self.data.index[-1]}" if hasattr(self.data, 'index') else "N/A"
            },
            'methods_summary': {}
        }
        
        for method, results in self.results.items():
            if 'anomaly_score' in results:
                scores = results['anomaly_score']
                report['methods_summary'][method] = {
                    'n_anomalies': results['n_anomalies'],
                    'anomaly_percentage': (results['n_anomalies'] / len(self.data)) * 100,
                    'max_score': np.max(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                }
        
        # Save report
        os.makedirs('results', exist_ok=True)
        
        with open('results/anomaly_detection_report.txt', 'w') as f:
            f.write("FLIGHT BLACK BOX ANOMALY DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"Shape: {report['dataset_info']['shape']}\n")
            f.write(f"Columns: {report['dataset_info']['columns']}\n")
            f.write(f"Time Range: {report['dataset_info']['time_range']}\n\n")
            
            f.write("ANOMALY DETECTION RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for method, summary in report['methods_summary'].items():
                f.write(f"\n{method}:\n")
                f.write(f"  Anomalies Found: {summary['n_anomalies']}\n")
                f.write(f"  Anomaly Percentage: {summary['anomaly_percentage']:.2f}%\n")
                f.write(f"  Max Score: {summary['max_score']:.4f}\n")
                f.write(f"  Mean Score: {summary['mean_score']:.4f}\n")
                f.write(f"  Std Score: {summary['std_score']:.4f}\n")
        
        # Save detailed results as CSV
        results_df = pd.DataFrame()
        for method, results in self.results.items():
            if 'anomaly_score' in results:
                results_df[f'{method}_score'] = pd.Series(results['anomaly_score'])
        
        # Add original data
        for col in self.data.columns:
            results_df[f'original_{col}'] = self.data[col].values
        
        results_df.to_csv('results/detailed_anomaly_scores.csv', index=False)
        
        print("Report saved to results/anomaly_detection_report.txt")
        print("Detailed scores saved to results/detailed_anomaly_scores.csv")
        
        return report
    
    def run_all_methods(self):
        """Run all anomaly detection methods"""
        print("Running all anomaly detection methods...")
        
        # Run each method
        self.dbscan_anomaly_detection()
        self.kmeans_anomaly_detection()
        self.isolation_forest_detection()
        self.pca_reconstruction_detection()
        self.lstm_autoencoder_detection()
        self.transformer_anomaly_detection()
        
        # Create ensemble
        self.ensemble_detection()
        
        # Generate visualizations and report
        self.visualize_results()
        self.generate_report()
        
        print("All methods completed successfully!")


def main():
    """Main function to run the anomaly detection system"""
    
    # Check for CSV file
    csv_files = [f for f in os.listdir('data/') if f.endswith('.csv')] if os.path.exists('data/') else []
    
    if not csv_files:
        print("No CSV files found in data/ directory!")
        print("Please place your flight black box CSV file in the data/ directory.")
        return
    
    # Use the first CSV file found
    csv_path = os.path.join('data', csv_files[0])
    print(f"Using CSV file: {csv_path}")
    
    # Create detector instance
    detector = FlightAnomalyDetector(csv_path)
    
    # Load data
    if not detector.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run all methods
    detector.run_all_methods()
    
    print("\nAnomaly detection completed!")
    print("Check the results/ directory for detailed outputs.")


if __name__ == "__main__":
    main()