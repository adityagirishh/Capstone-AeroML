#!/usr/bin/env python3
"""
Advanced Anomaly Detection Methods
Implements TranAD-inspired transformer and other sophisticated techniques
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TransformerBlock(nn.Module):
    """Transformer block for time series anomaly detection"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class TranADModel(nn.Module):
    """TranAD-inspired model for anomaly detection"""
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 3, seq_len: int = 100):
        super(TranADModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class AdvancedAnomalyDetector:
    """Advanced anomaly detection using sophisticated methods"""
    
    def __init__(self, data: np.ndarray, device: str = 'cpu'):
        self.data = data
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data)
        
    def create_sequences(self, data: np.ndarray, seq_len: int, step: int = 1):
        """Create overlapping sequences from time series data"""
        sequences = []
        for i in range(0, len(data) - seq_len + 1, step):
            sequences.append(data[i:i + seq_len])
        return np.array(sequences)
    
    def tranad_anomaly_detection(self, seq_len: int = 100, epochs: int = 100, 
                                lr: float = 0.001, batch_size: int = 32):
        """TranAD-inspired anomaly detection"""
        print("Running TranAD anomaly detection...")
        
        # Create sequences
        sequences = self.create_sequences(self.scaled_data, seq_len)
        
        # Convert to tensors
        train_data = torch.FloatTensor(sequences).to(self.device)
        train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = TranADModel(
            input_dim=self.scaled_data.shape[1],
            seq_len=seq_len
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                x = batch[0]
                optimizer.zero_grad()
                
                # Forward pass
                reconstruction = model(x)
                loss = criterion(reconstruction, x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Evaluation
        model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = torch.FloatTensor(sequences[i:i+batch_size]).to(self.device)
                reconstruction = model(batch)
                error = torch.mean((batch - reconstruction) ** 2, dim=(1, 2))
                reconstruction_errors.extend(error.cpu().numpy())
        
        # Create anomaly scores for full time series
        full_scores = np.zeros(len(self.data))
        for i, error in enumerate(reconstruction_errors):
            start_idx = i
            end_idx = min(start_idx + seq_len, len(full_scores))
            full_scores[start_idx:end_idx] = np.maximum(full_scores[start_idx:end_idx], error)
        
        # Normalize scores
        if np.max(full_scores) > 0:
            full_scores = full_scores / np.max(full_scores)
        
        print(f"TranAD training completed. Max reconstruction error: {np.max(reconstruction_errors):.6f}")
        
        return full_scores, train_losses
    
    def matrix_profile_anomaly_detection(self, window_size: int = 50):
        """Matrix Profile-based anomaly detection"""
        print("Running Matrix Profile anomaly detection...")
        
        try:
            import stumpy
            
            # For multivariate data, we'll use each dimension separately
            all_scores = []
            
            for dim in range(self.scaled_data.shape[1]):
                # Compute matrix profile
                mp = stumpy.stump(self.scaled_data[:, dim], m=window_size)
                distances = mp[:, 0]  # Matrix profile distances
                
                # Handle NaN values
                distances = np.nan_to_num(distances, nan=0)
                
                # Pad to match original length
                padded_distances = np.zeros(len(self.data))
                padded_distances[:len(distances)] = distances
                
                all_scores.append(padded_distances)
            
            # Combine scores across dimensions
            combined_scores = np.mean(all_scores, axis=0)
            
            # Normalize
            if np.max(combined_scores) > 0:
                combined_scores = combined_scores / np.max(combined_scores)
            
            print("Matrix Profile anomaly detection completed.")
            return combined_scores
            
        except ImportError:
            print("Warning: stumpy not available. Skipping Matrix Profile detection.")
            return np.zeros(len(self.data))
    
    def variational_autoencoder_detection(self, latent_dim: int = 32, seq_len: int = 100, 
                                        epochs: int = 50, lr: float = 0.001):
        """Variational Autoencoder for anomaly detection"""
        print("Running Variational Autoencoder anomaly detection...")
        
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim, seq_len):
                super(VAE, self).__init__()
                self.seq_len = seq_len
                self.input_dim = input_dim
                
                # Encoder
                self.encoder_lstm = nn.LSTM(input_dim, 64, batch_first=True)
                self.mu_layer = nn.Linear(64, latent_dim)
                self.logvar_layer = nn.Linear(64, latent_dim)
                
                # Decoder
                self.decoder_input = nn.Linear(latent_dim, 64)
                self.decoder_lstm = nn.LSTM(64, 64, batch_first=True)
                self.decoder_output = nn.Linear(64, input_dim)
                
            def encode(self, x):
                lstm_out, (h, c) = self.encoder_lstm(x)
                h = h[-1]  # Take last hidden state
                mu = self.mu_layer(h)
                logvar = self.logvar_layer(h)
                return mu, logvar
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                batch_size = z.size(0)
                z = self.decoder_input(z)
                z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
                lstm_out, _ = self.decoder_lstm(z)
                output = self.decoder_output(lstm_out)
                return output
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                reconstruction = self.decode(z)
                return reconstruction, mu, logvar
        
        # Create sequences
        sequences = self.create_sequences(self.scaled_data, seq_len)
        train_data = torch.FloatTensor(sequences).to(self.device)
        train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
        
        # Initialize VAE
        vae = VAE(self.scaled_data.shape[1], latent_dim, seq_len).to(self.device)
        optimizer = optim.Adam(vae.parameters(), lr=lr)
        
        def vae_loss(recon_x, x, mu, logvar):
            mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return mse + kld
        
        # Training
        vae.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                x = batch[0]
                optimizer.zero_grad()
                
                recon_x, mu, logvar = vae(x)
                loss = vae_loss(recon_x, x, mu, logvar)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"VAE Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
        
        # Evaluation
        vae.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), 32):
                batch = torch.FloatTensor(sequences[i:i+32]).to(self.device)
                recon_batch, _, _ = vae(batch)
                error = torch.mean((batch - recon_batch) ** 2, dim=(1, 2))
                reconstruction_errors.extend(error.cpu().numpy())
        
        # Create full scores
        full_scores = np.zeros(len(self.data))
        for i, error in enumerate(reconstruction_errors):
            start_idx = i
            end_idx = min(start_idx + seq_len, len(full_scores))
            full_scores[start_idx:end_idx] = np.maximum(full_scores[start_idx:end_idx], error)
        
        # Normalize
        if np.max(full_scores) > 0:
            full_scores = full_scores / np.max(full_scores)
        
        print("VAE anomaly detection completed.")
        return full_scores
    
    def spectral_residual_detection(self, window_size: int = 100):
        """Spectral Residual anomaly detection"""
        print("Running Spectral Residual anomaly detection...")
        
        def spectral_residual_transform(x):
            """Apply spectral residual transform to detect anomalies"""
            # FFT
            fft_x = np.fft.fft(x)
            magnitude = np.abs(fft_x)
            phase = np.angle(fft_x)
            
            # Log magnitude
            log_magnitude = np.log(magnitude + 1e-8)
            
            # Spectral residual
            kernel = np.ones(3) / 3  # Simple averaging filter
            if len(log_magnitude) >= 3:
                smoothed = np.convolve(log_magnitude, kernel, mode='same')
                residual = log_magnitude - smoothed
            else:
                residual = log_magnitude
            
            # Inverse FFT
            ifft_result = np.fft.ifft(np.exp(residual + 1j * phase))
            saliency = np.abs(ifft_result) ** 2
            
            return saliency
        
        all_scores = []
        
        # Apply to each dimension
        for dim in range(self.scaled_data.shape[1]):
            dim_data = self.scaled_data[:, dim]
            
            # Apply spectral residual in sliding windows
            scores = np.zeros(len(dim_data))
            
            for i in range(0, len(dim_data) - window_size + 1, window_size // 2):
                window = dim_data[i:i + window_size]
                saliency = spectral_residual_transform(window)
                
                # Average saliency for the window
                avg_saliency = np.mean(saliency)
                scores[i:i + window_size] = np.maximum(scores[i:i + window_size], avg_saliency)
            
            all_scores.append(scores)
        
        # Combine scores
        combined_scores = np.mean(all_scores, axis=0)
        
        # Normalize
        if np.max(combined_scores) > 0:
            combined_scores = combined_scores / np.max(combined_scores)
        
        print("Spectral Residual anomaly detection completed.")
        return combined_scores
    
    def statistical_anomaly_detection(self, window_size: int = 50, n_sigma: float = 3):
        """Statistical anomaly detection using multiple statistical tests"""
        print("Running Statistical anomaly detection...")
        
        def calculate_statistical_features(window):
            """Calculate various statistical features for a window"""
            features = {}
            
            # Basic statistics
            features['mean'] = np.mean(window, axis=0)
            features['std'] = np.std(window, axis=0)
            features['skew'] = np.mean(((window - features['mean']) / features['std']) ** 3, axis=0)
            features['kurtosis'] = np.mean(((window - features['mean']) / features['std']) ** 4, axis=0) - 3
            
            # Range and IQR
            features['range'] = np.max(window, axis=0) - np.min(window, axis=0)
            features['iqr'] = np.percentile(window, 75, axis=0) - np.percentile(window, 25, axis=0)
            
            # Autocorrelation (lag 1)
            features['autocorr'] = []
            for dim in range(window.shape[1]):
                if len(window) > 1:
                    corr = np.corrcoef(window[:-1, dim], window[1:, dim])[0, 1]
                    features['autocorr'].append(corr if not np.isnan(corr) else 0)
                else:
                    features['autocorr'].append(0)
            features['autocorr'] = np.array(features['autocorr'])
            
            return features
        
        # Calculate rolling statistics
        all_features = []
        
        for i in range(len(self.scaled_data) - window_size + 1):
            window = self.scaled_data[i:i + window_size]
            features = calculate_statistical_features(window)
            all_features.append(features)
        
        # Convert to arrays
        feature_names = ['mean', 'std', 'skew', 'kurtosis', 'range', 'iqr', 'autocorr']
        feature_arrays = {}
        
        for name in feature_names:
            feature_arrays[name] = np.array([f[name] for f in all_features])
        
        # Detect anomalies using z-score
        anomaly_scores = np.zeros((len(all_features), self.scaled_data.shape[1]))
        
        for name, feature_array in feature_arrays.items():
            # Calculate z-scores
            mean_feature = np.mean(feature_array, axis=0)
            std_feature = np.std(feature_array, axis=0)
            std_feature = np.where(std_feature == 0, 1, std_feature)  # Avoid division by zero
            
            z_scores = np.abs((feature_array - mean_feature) / std_feature)
            anomaly_indicator = z_scores > n_sigma
            
            anomaly_scores += anomaly_indicator
        
        # Aggregate across dimensions and features
        final_scores = np.mean(anomaly_scores, axis=1)
        
        # Pad to original length
        full_scores = np.zeros(len(self.data))
        full_scores[window_size//2:-window_size//2+1] = final_scores
        
        # Normalize
        if np.max(full_scores) > 0:
            full_scores = full_scores / np.max(full_scores)
        
        print("Statistical anomaly detection completed.")
        return full_scores
    
    def run_all_advanced_methods(self, seq_len: int = 100):
        """Run all advanced anomaly detection methods"""
        results = {}
        
        print("Running all advanced anomaly detection methods...")
        
        # TranAD
        try:
            tranad_scores, _ = self.tranad_anomaly_detection(seq_len=seq_len, epochs=50)
            results['TranAD'] = tranad_scores
        except Exception as e:
            print(f"TranAD failed: {e}")
            results['TranAD'] = np.zeros(len(self.data))
        
        # Matrix Profile
        try:
            mp_scores = self.matrix_profile_anomaly_detection()
            results['MatrixProfile'] = mp_scores
        except Exception as e:
            print(f"Matrix Profile failed: {e}")
            results['MatrixProfile'] = np.zeros(len(self.data))
        
        # VAE
        try:
            vae_scores = self.variational_autoencoder_detection(seq_len=seq_len, epochs=30)
            results['VAE'] = vae_scores
        except Exception as e:
            print(f"VAE failed: {e}")
            results['VAE'] = np.zeros(len(self.data))
        
        # Spectral Residual
        try:
            sr_scores = self.spectral_residual_detection()
            results['SpectralResidual'] = sr_scores
        except Exception as e:
            print(f"Spectral Residual failed: {e}")
            results['SpectralResidual'] = np.zeros(len(self.data))
        
        # Statistical
        try:
            stat_scores = self.statistical_anomaly_detection()
            results['Statistical'] = stat_scores
        except Exception as e:
            print(f"Statistical detection failed: {e}")
            results['Statistical'] = np.zeros(len(self.data))
        
        return results


def integrate_advanced_methods(main_detector, csv_path: str):
    """Integrate advanced methods with the main detector"""
    
    print("Integrating advanced anomaly detection methods...")
    
    # Load data for advanced methods
    data = pd.read_csv(csv_path)
    
    # Handle time column if present
    time_cols = [col for col in data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
    if time_cols:
        data[time_cols[0]] = pd.to_datetime(data[time_cols[0]], errors='coerce')
        data = data.set_index(time_cols[0])
    
    # Select numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_cols]
    
    # Handle missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize advanced detector
    advanced_detector = AdvancedAnomalyDetector(data.values, device=device)
    
    # Run advanced methods
    advanced_results = advanced_detector.run_all_advanced_methods(seq_len=min(100, len(data)//10))
    
    # Add results to main detector
    for method_name, scores in advanced_results.items():
        main_detector.results[f'Advanced_{method_name}'] = {
            'anomaly_score': scores,
            'n_anomalies': np.sum(scores > 0.5)
        }
    
    print("Advanced methods integration completed!")
    
    return advanced_results


if __name__ == "__main__":
    # Example usage
    print("Advanced Anomaly Detection Methods")
    print("This module provides sophisticated anomaly detection algorithms")
    print("Import this module and use integrate_advanced_methods() function")