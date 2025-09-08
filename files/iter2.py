# ==============================================================================
# TIME-SERIES SPECIFIC UNSUPERVISED METHODS FOR FLIGHT MANEUVER PREDICTION
# ==============================================================================
# This implementation combines:
# 1. LSTM Autoencoders for dimensionality reduction and embeddings
# 2. Clustering (K-means & DBSCAN) on learned embeddings
# 3. Hidden Markov Models (HMMs) for sequence modeling and prediction
# ==============================================================================

# --- Library Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# STEP 1: LOAD AND PREPARE FLIGHT DATA
# ==============================================================================
print("=" * 60)
print("STEP 1: LOADING AND PREPARING FLIGHT DATA")
print("=" * 60)

# --- Configuration ---
file_path = '/Users/adityagirish/Desktop/AerX shared data /AerX(for our understanding) .csv'

# --- Load Data ---
try:
    # Skip metadata lines (first 3 lines contain headers and comments)
    data = pd.read_csv(file_path, low_memory=False, skiprows=3)
    print(f"✓ Successfully loaded {file_path}")
    print(f"  Data shape: {data.shape}")
except FileNotFoundError:
    print(f"✗ Error: The file '{file_path}' was not found.")
    exit()

# --- Column Cleaning ---
data.columns = data.columns.str.strip()
column_mapping = {
    'Lcl Date': 'lcl date', 'Lcl Time': 'lcl time', 'UTCOfst': 'utcofst',
    'AtvWpt': 'atvwpt', 'Latitude': 'latitude', 'Longitude': 'longitude',
    'AltInd': 'altind', 'BaroA': 'baroa', 'AltMSL': 'altmsl', 'OAT': 'oat',
    'IAS': 'ias', 'GndSpd': 'gndspd', 'VSpd': 'vspd', 'Pitch': 'pitch',
    'Roll': 'roll', 'LatAc': 'latac', 'NormAc': 'normac', 'HDG': 'hdg',
    'TRK': 'trk', 'volt1': 'volt1', 'volt2': 'volt2', 'amp1': 'amp1',
    'amp2': 'amp2', 'FQtyL': 'fqtyl', 'FQtyR': 'fqtyr', 'E1 FFlow': 'e1 fflow',
    'E1 OilT': 'e1 oilt', 'E1 OilP': 'e1 oilp', 'E1 RPM': 'e1 rpm',
    'E1 CHT1': 'e1 cht1', 'E1 CHT2': 'e1 cht2', 'E1 CHT3': 'e1 cht3',
    'E1 CHT4': 'e1 cht4', 'E1 EGT1': 'e1 egt1', 'E1 EGT2': 'e1 egt2',
    'E1 EGT3': 'e1 egt3', 'E1 EGT4': 'e1 egt4', 'AltGPS': 'altgps', 'TAS': 'tas',
    'HSIS': 'hsis', 'CRS': 'crs', 'NAV1': 'nav1', 'NAV2': 'nav2',
    'COM1': 'com1', 'COM2': 'com2', 'HCDI': 'hcdi', 'VCDI': 'vcdi',
    'WndSpd': 'wndspd', 'WndDr': 'wnddr', 'WptDst': 'wptdst', 'WptBrg': 'wptbrg',
    'MagVar': 'magvar', 'AfcsOn': 'afcson', 'RollM': 'rollm', 'PitchM': 'pitchm',
    'RollC': 'rollc', 'PichC': 'pichc', 'VSpdG': 'vspdg', 'GPSfix': 'gpsfix',
    'HAL': 'hal', 'VAL': 'val', 'HPLwas': 'hplwas', 'HPLfd': 'hplfd',
    'VPLwas': 'vplwas'
}
data.rename(columns=lambda c: column_mapping.get(c, c), inplace=True)
print("✓ Cleaned column headers")

# --- Feature Selection ---
# Select key kinematic and engine features as recommended
key_features = [
    'altind', 'altmsl', 'altgps', 'ias', 'gndspd', 'tas', 'vspd', 'vspdg',
    'pitch', 'roll', 'hdg', 'e1 rpm', 'e1 oilt', 'e1 cht1', 'e1 cht2', 
    'e1 cht3', 'e1 cht4', 'e1 egt1', 'e1 egt2', 'e1 egt3', 'e1 egt4',
    'afcson', 'rollm', 'pitchm', 'rollc', 'pichc', 'gpsfix', 'hal', 'val', 
    'hplwas', 'hplfd', 'vplwas', 'fqtyl', 'fqtyr', 'volt1', 'volt2', 'amp1', 'amp2'
]

# Filter to selected features and handle missing values
df = data[key_features].copy()
df = df.dropna()
print(f"✓ Selected {len(key_features)} key features")
print(f"  Filtered data shape: {df.shape}")

# --- Data Statistics ---
print("\n--- Data Statistics ---")
print(df.describe())

# --- Normalization ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df.values)
print("✓ Applied StandardScaler normalization")

# ==============================================================================
# STEP 2: CREATE TIME-SERIES WINDOWS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 2: CREATING TIME-SERIES WINDOWS")
print("=" * 60)

def create_windows(data, window_size=10, step_size=1):
    """
    Create sliding windows from time-series data
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        window_size: size of each window
        step_size: step between windows
    
    Returns:
        numpy array of shape (n_windows, window_size, n_features)
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i+window_size])
    return np.array(windows)

# Create windows with size 10 (matching your 5-second intervals)
window_size = 10
X_windows = create_windows(data_scaled, window_size=window_size)
print(f"✓ Created {X_windows.shape[0]} windows of size {window_size}")
print(f"  Window shape: {X_windows.shape}")

# ==============================================================================
# STEP 3: LSTM AUTOENCODER FOR EMBEDDINGS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: TRAINING LSTM AUTOENCODER")
print("=" * 60)

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series dimensionality reduction
    """
    def __init__(self, input_dim, hidden_dim=64, embed_dim=32):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_dim, embed_dim)
        
        # Decoder  
        self.decoder_linear = nn.Linear(embed_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encoding
        _, (h, _) = self.encoder_lstm(x)
        embedding = self.encoder_linear(h.squeeze(0))
        
        # Decoding
        embed_repeated = embedding.unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_input = self.decoder_linear(embed_repeated)
        out, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.decoder_output(out)
        
        return reconstruction, embedding
    
    def get_embeddings(self, x):
        """Extract embeddings without reconstruction"""
        _, (h, _) = self.encoder_lstm(x)
        return self.encoder_linear(h.squeeze(0))

# --- Model Configuration ---
input_dim = len(key_features)  # Number of features
hidden_dim = 64
embed_dim = 32
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# --- Initialize Model ---
model = LSTMAutoencoder(input_dim, hidden_dim, embed_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print(f"✓ Model initialized:")
print(f"  Input dim: {input_dim}, Hidden dim: {hidden_dim}, Embed dim: {embed_dim}")

# --- Data Preparation for Training ---
dataset = TensorDataset(torch.tensor(X_windows, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Training Loop ---
print("\n--- Training Autoencoder ---")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in loader:
        inputs = batch[0]
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / len(loader)
        print(".4f")

print("✓ Autoencoder training completed")

# --- Extract Embeddings ---
print("\n--- Extracting Embeddings ---")
model.eval()
with torch.no_grad():
    embeddings = model.get_embeddings(torch.tensor(X_windows, dtype=torch.float32))
    embeddings = embeddings.numpy()

print(f"✓ Extracted embeddings shape: {embeddings.shape}")

# ==============================================================================
# STEP 4: CLUSTERING ON EMBEDDINGS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4: CLUSTERING ON EMBEDDINGS")
print("=" * 60)

def evaluate_clustering(embeddings, labels, method_name):
    """Evaluate clustering quality"""
    try:
        silhouette = silhouette_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)
        print(f"✓ {method_name} Evaluation:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.4f}")
        return silhouette, calinski
    except:
        print(f"✓ {method_name} clustering completed")
        return None, None

# --- K-means Clustering ---
print("\n--- K-means Clustering ---")
n_clusters_range = range(2, 8)  # Test different numbers of clusters
best_silhouette = -1
best_k = 2

for k in n_clusters_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, labels_temp)
    
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

print(f"✓ Best K-means with k={best_k}")

# Apply best K-means
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(embeddings)
evaluate_clustering(embeddings, kmeans_labels, "K-means")

# --- DBSCAN Clustering ---
print("\n--- DBSCAN Clustering ---")
# Use PCA to reduce dimensionality for better DBSCAN performance
pca = PCA(n_components=10)
embeddings_pca = pca.fit_transform(embeddings)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(embeddings_pca)
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"✓ DBSCAN found {n_dbscan_clusters} clusters")

if n_dbscan_clusters > 1:
    evaluate_clustering(embeddings_pca, dbscan_labels, "DBSCAN")

# ==============================================================================
# STEP 5: HIDDEN MARKOV MODEL FOR SEQUENCE MODELING
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 5: HIDDEN MARKOV MODEL FOR SEQUENCE MODELING")
print("=" * 60)

try:
    from hmmlearn import hmm
    
    # --- Prepare sequence data ---
    # Use the scaled data for HMM (not embeddings, as HMM works on observations)
    sequence_data = data_scaled
    
    # --- Fit HMM ---
    n_components = best_k  # Use same number as K-means clusters
    hmm_model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type='diag', 
        n_iter=100,
        random_state=42
    )
    
    print(f"✓ Fitting HMM with {n_components} hidden states...")
    hmm_model.fit(sequence_data)
    
    # --- Decode hidden states ---
    hidden_states = hmm_model.predict(sequence_data)
    print(f"✓ HMM decoded {len(set(hidden_states))} unique states")
    
    # --- Analyze transition probabilities ---
    print("\n--- HMM Transition Matrix ---")
    transition_matrix = hmm_model.transmat_
    print("Transition probabilities:")
    for i in range(n_components):
        for j in range(n_components):
            print(".3f")
    
    # --- State means analysis ---
    print("\n--- HMM State Analysis ---")
    state_means = hmm_model.means_
    feature_names = df.columns
    
    for state in range(n_components):
        print(f"\nState {state} characteristics:")
        for feat_idx, feature in enumerate(feature_names):
            mean_val = state_means[state, feat_idx]
            print(".2f")
    
except ImportError:
    print("✗ hmmlearn not available. Install with: pip install hmmlearn")
    hidden_states = None

# ==============================================================================
# STEP 6: VISUALIZATION AND ANALYSIS
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 6: VISUALIZATION AND ANALYSIS")
print("=" * 60)

# --- Plot 1: Embeddings with Clusters ---
plt.figure(figsize=(15, 10))

# 2D PCA of embeddings
pca_2d = PCA(n_components=2)
embeddings_2d = pca_2d.fit_transform(embeddings)

plt.subplot(2, 3, 1)
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.title('K-means Clusters on Embeddings (2D PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter)

# --- Plot 2: Cluster Time Series ---
plt.subplot(2, 3, 2)
time_indices = np.arange(len(kmeans_labels))
plt.scatter(time_indices, kmeans_labels, alpha=0.6, s=2)
plt.title('K-means Clusters Over Time')
plt.xlabel('Time Window Index')
plt.ylabel('Cluster Label')
plt.ylim(-0.5, best_k - 0.5)

# --- Plot 3: Feature Distributions by Cluster ---
plt.subplot(2, 3, 3)
# Plot altitude distribution by cluster
for cluster in range(best_k):
    cluster_mask = kmeans_labels == cluster
    cluster_windows = X_windows[cluster_mask]
    if len(cluster_windows) > 0:
        # Use altitude (first feature in key_features which is 'altind')
        alt_values = cluster_windows[:, :, 0].flatten()
        plt.hist(alt_values, alpha=0.5, bins=20, label=f'Cluster {cluster}')
plt.title('Altitude Distribution by Cluster')
plt.xlabel('Scaled Altitude')
plt.ylabel('Frequency')
plt.legend()

# --- Plot 4: HMM States Over Time ---
if hidden_states is not None:
    plt.subplot(2, 3, 4)
    plt.plot(hidden_states[:500], alpha=0.7)  # First 500 points
    plt.title('HMM Hidden States Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Hidden State')
    plt.ylim(-0.5, n_components - 0.5)

# --- Plot 5: Vertical Speed Patterns ---
plt.subplot(2, 3, 5)
for cluster in range(best_k):
    cluster_mask = kmeans_labels == cluster
    cluster_windows = X_windows[cluster_mask]
    if len(cluster_windows) > 0:
        # Vertical speed is the 7th feature (index 6) in key_features
        vspd_values = cluster_windows[:, :, 6].flatten()
        plt.hist(vspd_values, alpha=0.5, bins=20, label=f'Cluster {cluster}')
plt.title('Vertical Speed Distribution by Cluster')
plt.xlabel('Scaled Vertical Speed')
plt.ylabel('Frequency')
plt.legend()

# --- Plot 6: Maneuver Identification ---
plt.subplot(2, 3, 6)
# Plot key flight parameters over time with cluster coloring
time_range = range(min(200, len(df)))  # First 200 points
colors = plt.cm.viridis(np.linspace(0, 1, best_k))

# Map clusters to time windows
window_centers = np.arange(window_size//2, len(kmeans_labels) + window_size//2, 1)[:len(kmeans_labels)]

for cluster in range(best_k):
    cluster_times = window_centers[kmeans_labels == cluster]
    cluster_times = cluster_times[cluster_times < len(df)]
    if len(cluster_times) > 0:
        # Use altitude (altind is the first feature)
        plt.scatter(cluster_times, df.iloc[cluster_times]['altind'], 
                   c=[colors[cluster]], alpha=0.6, s=3, label=f'Cluster {cluster}')

plt.title('Altitude with Cluster Coloring')
plt.xlabel('Time Index')
plt.ylabel('Altitude (ft)')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# STEP 7: MANEUVER INTERPRETATION AND SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 7: MANEUVER INTERPRETATION AND SUMMARY")
print("=" * 60)

# --- Cluster Analysis ---
print("\n--- Cluster Analysis ---")
for cluster in range(best_k):
    cluster_mask = kmeans_labels == cluster
    cluster_count = np.sum(cluster_mask)
    cluster_percentage = (cluster_count / len(kmeans_labels)) * 100
    
    print(f"\nCluster {cluster}:")
    print(f"  Count: {cluster_count} windows ({cluster_percentage:.1f}%)")
    
    # Analyze mean feature values for this cluster
    cluster_windows = X_windows[cluster_mask]
    if len(cluster_windows) > 0:
        mean_values = np.mean(cluster_windows.reshape(-1, len(key_features)), axis=0)
        print(f"  Mean feature values:")
        for i, feature in enumerate(key_features):
            print(".2f")

# --- Maneuver Classification ---
print("\n--- Potential Maneuver Classification ---")
print("Based on cluster characteristics:")

for cluster in range(best_k):
    cluster_mask = kmeans_labels == cluster
    cluster_windows = X_windows[cluster_mask]
    
    if len(cluster_windows) > 0:
        # Analyze key indicators (corrected indices based on key_features order)
        mean_alt = np.mean(cluster_windows[:, :, 0])      # altind
        mean_vspd = np.mean(cluster_windows[:, :, 6])     # vspd (7th feature)
        mean_pitch = np.mean(cluster_windows[:, :, 8])    # pitch (9th feature)
        mean_ias = np.mean(cluster_windows[:, :, 3])      # ias (4th feature)
        
        # Classify maneuver based on patterns
        if mean_vspd > 0.5 and mean_pitch > 0.2:
            maneuver = "Climb"
        elif mean_vspd < -0.5 and mean_pitch < -0.2:
            maneuver = "Descent"
        elif mean_ias > 0.5 and abs(mean_vspd) < 0.3:
            maneuver = "Cruise/Acceleration"
        elif mean_ias < -0.5 and abs(mean_vspd) < 0.3:
            maneuver = "Deceleration"
        elif abs(mean_vspd) < 0.2 and abs(mean_ias) < 0.2:
            maneuver = "Ground/Taxi"
        else:
            maneuver = "Transition"
            
        print(f"  Cluster {cluster}: {maneuver}")
        print(".2f")
        print(".2f")
        print(".2f")

# --- Performance Summary ---
print("\n--- Performance Summary ---")
print("✓ Data processing completed successfully")
print(f"✓ Processed {len(df)} data points into {len(X_windows)} windows")
print(f"✓ Trained LSTM Autoencoder with {embed_dim}-dimensional embeddings")
print(f"✓ K-means clustering identified {best_k} maneuver patterns")
if hidden_states is not None:
    print(f"✓ HMM modeling with {n_components} hidden states completed")
print("✓ Visualization and analysis completed")

print("\n" + "=" * 60)
print("IMPLEMENTATION COMPLETE")
print("=" * 60)
print("\nNext steps:")
print("1. Tune hyperparameters (window size, embedding dimensions, clusters)")
print("2. Validate with additional flight logs")
print("3. Implement real-time prediction pipeline")
print("4. Add anomaly detection capabilities")
print("5. Integrate with flight control systems")