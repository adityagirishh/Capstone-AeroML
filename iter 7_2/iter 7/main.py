import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import os
from scipy.signal import medfilt  # ✅ ADDED FOR MEDIAN FILTER

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ========================
# Step 1: Data Preparation
# ========================

file_path = "/Users/adityagirish/capstoned/sample run/data/sample_flight_data.csv"
df = pd.read_csv(file_path)

# Define key features corpus — UPDATED WITH CORRECT COLUMN NAMES
key_feature_corpus = [
    'latitude', 'longitude', 'altmsl',  # ✅ Corrected names
    'altind', 'altgps', 'ias', 'gndspd', 'tas', 'vspd', 'vspdg',
    'pitch', 'roll', 'hdg', 'e1 rpm', 'e1 oilt', 'e1 cht1', 'e1 cht2',
    'e1 cht3', 'e1 cht4', 'e1 egt1', 'e1 egt2', 'e1 egt3', 'e1 egt4',
    'afcson', 'rollm', 'pitchm', 'rollc', 'pichc', 'gpsfix', 'hal', 'val',
    'hplwas', 'hplfd', 'vplwas', 'fqtyl', 'fqtyr', 'volt1', 'volt2', 'amp1', 'amp2'
]

# Filter available features
key_features = [col for col in df.columns if col in key_feature_corpus]
print(f"Selected {len(key_features)} key features: {key_features}")

# ✅ Normalize ALL features together (including lat/long/altmsl)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[key_features])  # Shape: (50, N)

# Create sliding windows — smaller window for better resolution
def create_windows(data, window_size=10, step=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)  # Shape: (n_windows, window_size, N)

X = create_windows(data_scaled)  # (e.g., 46 windows if step=1, window=5)


# =============================
# Step 2: LSTM Autoencoder
# =============================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout_rate):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bottleneck = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encoder
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0)  # (batch, hidden_dim)
        embed = self.bottleneck(h)
        embed = self.dropout(embed)  # (batch, embed_dim)

        # Decoder: Repeat embedding to match sequence length
        embed_repeated = embed.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, embed_dim)
        out, _ = self.decoder(embed_repeated)  # (batch, seq_len, hidden_dim)
        reconstructed = self.output_layer(out)  # (batch, seq_len, input_dim)
        return reconstructed

    def encode(self, x):
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0)
        embed = self.bottleneck(h)
        return embed


# Instantiate model
input_dim = len(key_features)
model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64, embed_dim=16, dropout_rate=0.2)

# Optimizer and loss
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# DataLoader
X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train
print("Training LSTM Autoencoder...")
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in loader:
        inputs = batch[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {total_loss / len(loader):.6f}')

# Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model.encode(X_tensor).numpy()  # Shape: (n_windows, 16)

# ===================================================================
# Step 3: Find Optimal HMM Components via BIC
# ===================================================================
print("\nFinding optimal number of HMM states using BIC...")

# Set a range of possible component counts to test
n_components_range = range(2, 11)
bic_scores = []
hmm_models = []

for n in n_components_range:
    try:
        model_temp = hmm.GaussianHMM(n_components=n, covariance_type='diag', n_iter=100, random_state=42)
        model_temp.fit(data_scaled)
        
        # Calculate BIC: -2 * log-likelihood + n_params * log(n_samples)
        log_likelihood = model_temp.score(data_scaled)
        n_features = data_scaled.shape[1]
        
        # Number of parameters in a Gaussian HMM
        n_params = (n * (n - 1)) + (2 * n * n_features) # transmat + (means + covars)
        
        bic = -2 * log_likelihood + n_params * np.log(len(data_scaled))
        
        bic_scores.append(bic)
        hmm_models.append(model_temp)
        print(f"  - Components: {n}, Log-Likelihood: {log_likelihood:.2f}, BIC: {bic:.2f}")
    except Exception as e:
        print(f"  - Components: {n}, Failed with error: {e}")
        bic_scores.append(np.inf) # Penalize failures
        hmm_models.append(None)


# Find the best number of components
best_n_components = n_components_range[np.argmin(bic_scores)]
hmm_model = hmm_models[np.argmin(bic_scores)]

print(f"\n✅ Optimal number of components found: {best_n_components} (Lowest BIC)")


# Plot BIC scores
output_dir = "results-AeroML"
os.makedirs(output_dir, exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o', linestyle='-', color='indigo')
plt.title('Figure 0: HMM BIC Score vs. Number of Components', fontsize=14, fontweight='bold')
plt.xlabel('Number of Components (States)', fontsize=12)
plt.ylabel('Bayesian Information Criterion (BIC)', fontsize=12)
plt.xticks(n_components_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, "hmm_bic_scores.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ BIC score plot saved to '{output_dir}/hmm_bic_scores.png'")


# ===================================================================
# Step 4: Final HMM Modeling & Prediction — WITH MEDIAN FILTER
# ===================================================================

predicted_states = hmm_model.predict(data_scaled)  # Shape: (50,) — one per timestep!

# ✅ APPLY MEDIAN FILTER TO SMOOTH STATE TRANSITIONS
smoothed_states = medfilt(predicted_states, kernel_size=5).astype(int)
predicted_states = smoothed_states  # Use smoothed version for all downstream tasks

print("\nPredicted HMM States for Each Timestep (Smoothed):")
print(predicted_states)

score = hmm_model.score(data_scaled)
print(f"HMM Score (Log-Likelihood): {score:.4f}")

# Greedy next state
current_state = predicted_states[-1]
transition_probs = hmm_model.transmat_[current_state]
next_state_greedy = np.argmax(transition_probs)
print(f"\nMost Likely Next State (Greedy Prediction): {next_state_greedy}")

# Probabilistic forecasting
n_forecasts = 1000
possible_states = np.arange(hmm_model.n_components)
forecasted_states = np.random.choice(possible_states, size=n_forecasts, p=transition_probs)
state_counts = np.bincount(forecasted_states, minlength=hmm_model.n_components)

print("\n--- Probabilistic Forecast for the Next Timestep ---")
for i, count in enumerate(state_counts):
    probability = (count / n_forecasts) * 100
    print(f" - State {i}: ~{probability:.1f}% ({count} occurrences)")


# ==============================
# Step 5: Visualization & Export
# ==============================

def align_window_labels_to_timesteps(labels, window_size, total_timesteps):
    """
    Assigns each window's label to all timesteps it covers.
    Resolves overlaps by taking the mode (most frequent label) per timestep.
    """
    from scipy.stats import mode
    votes = [[] for _ in range(total_timesteps)]

    for win_start, label in enumerate(labels):
        for offset in range(window_size):
            t = win_start + offset
            if t < total_timesteps:
                votes[t].append(label)

    aligned = np.full(total_timesteps, -1, dtype=int)
    for t in range(total_timesteps):
        if len(votes[t]) > 0:
            aligned[t] = mode(votes[t], keepdims=False).mode

    return aligned.astype(float)  # Convert to float for NaN handling later


def visualise(autoencoder_model, hmm_model, X_tensor, data_scaled,
              predicted_states, key_features, original_df, window_size=5):
    """
    Generates comprehensive visualizations for research paper.
    NOW INCLUDES 3D FLIGHT PATH WITH HMM STATE OVERLAY.
    """
    print("Generating visualizations for the research paper...")
    output_dir = "results-AeroML"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. Reconstruction Error ---
    autoencoder_model.eval()
    with torch.no_grad():
        reconstructions = autoencoder_model(X_tensor)
        errors = torch.mean((X_tensor - reconstructions)**2, dim=(1, 2)).numpy()

    plt.figure(figsize=(14, 5))
    plt.plot(errors, label='Reconstruction Error', color='crimson', linewidth=2)
    plt.title('Figure 1: Autoencoder Reconstruction Error per Window', fontsize=14, fontweight='bold')
    plt.xlabel('Window Index', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "reconstruction_error.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 2. PCA of Embeddings (optional) ---
    with torch.no_grad():
        embeddings = autoencoder_model.encode(X_tensor).numpy()
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    # Dummy cluster labels for PCA (use HMM states aligned to windows)
    window_labels = []
    for i in range(len(embeddings)):
        # Take mode of HMM states in window i
        window_start = i
        window_end = min(i + window_size, len(predicted_states))
        window_states = predicted_states[window_start:window_end]
        from scipy.stats import mode
        label = mode(window_states, keepdims=False).mode if len(window_states) > 0 else -1
        window_labels.append(label)
    window_labels = np.array(window_labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=window_labels, cmap='viridis', s=80)
    plt.title('Figure 2: PCA Visualization of Embeddings (Colored by HMM State)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    if len(np.unique(window_labels)) > 1:
        legend = plt.legend(*scatter.legend_elements(), title="HMM States", fontsize=11)
        plt.gca().add_artist(legend)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "pca_embeddings_hmm.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3. HMM Transition Matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(hmm_model.transmat_, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, annot_kws={"size": 12})
    plt.title('Figure 3: HMM State Transition Probability Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('To State', fontsize=12)
    plt.ylabel('From State', fontsize=12)
    plt.savefig(os.path.join(output_dir, "hmm_transition_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4. Maneuver Segmentation Overlay ---
    # Since we're using HMM states, just align them (no voting needed — already per timestep)
    aligned_labels = predicted_states.astype(float)  # Already aligned to timesteps

    fig, ax1 = plt.subplots(figsize=(18, 7))

    # Primary axis: Flight parameters
    ax1.plot(original_df.index, original_df['altmsl'], label='Altitude (MSL)', color='blue', linewidth=2.5)
    ax1.plot(original_df.index, original_df['ias'] * 10, label='Indicated Airspeed (x10)', color='green', linewidth=2, linestyle='--')
    ax1.set_xlabel('Time (Timestep)', fontsize=13)
    ax1.set_ylabel('Altitude / Airspeed', fontsize=13, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Maneuver labels
    ax2 = ax1.twinx()
    ax2.plot(original_df.index, aligned_labels, label='HMM State (Maneuver)', color='red', marker='o', linestyle='-', linewidth=2, markersize=6)
    ax2.set_ylabel('HMM State', fontsize=13, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.5, np.nanmax(aligned_labels) + 0.5)
    ax2.legend(loc='upper right', fontsize=12)

    plt.title('Figure 4: HMM State Segments Overlaid on Flight Parameters', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "hmm_segmentation_overlay.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 5. 3D Flight Path with HMM State Coloring ---
    if 'latitude' in original_df.columns and 'longitude' in original_df.columns and 'altmsl' in original_df.columns:
        print("✅ Plotting 3D Flight Path with HMM State Coloring...")

        from mpl_toolkits.mplot3d import Axes3D

        # Get unique HMM states
        unique_states = np.unique(predicted_states)
        n_states = len(unique_states)

        # Generate random but distinct colors
        np.random.seed(42)
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, max(n_states, 10)))
        np.random.shuffle(colors)
        state_color_map = {int(state): colors[i % len(colors)] for i, state in enumerate(unique_states)}

        # Prepare data
        lat = original_df['latitude'].values
        long = original_df['longitude'].values
        alt = original_df['altmsl'].values
        timesteps = np.arange(len(lat))

        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each segment with its HMM state color
        for i in range(len(timesteps) - 1):
            state = predicted_states[i]
            color = state_color_map.get(int(state), 'gray')
            ax.plot(
                long[i:i+2], lat[i:i+2], alt[i:i+2],
                color=color, linewidth=3, alpha=0.8
            )

        ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
        ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
        ax.set_zlabel('Altitude (MSL)', fontsize=12, labelpad=10)
        ax.set_title('Figure 5: 3D Flight Path Colored by HMM State (Maneuver)', fontsize=16, fontweight='bold', pad=20)

        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color, lw=4, label=f'State {int(state)}')
            for state, color in state_color_map.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=11)

        # Improve viewing angle
        ax.view_init(elev=20, azim=-60)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "flight_path_3d_hmm.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 3D Flight Path saved to 'results-AeroML/flight_path_3d_hmm.png'")
    else:
        print("⚠️  Skipping 3D Flight Path: Columns 'latitude', 'longitude', or 'atmsl' not found in data.")

    # --- 6. Save Segmented Data ---
    segmented_df = original_df.copy()
    segmented_df['hmm_state'] = predicted_states  # Final maneuver label

    csv_path = os.path.join(output_dir, "segmented_flight_data.csv")
    segmented_df.to_csv(csv_path, index=False)
    print(f"✅ Visualizations and segmented data saved to '{output_dir}/'")


# Run visualization — using HMM states for labeling
visualise(
    autoencoder_model=model,
    hmm_model=hmm_model,
    X_tensor=X_tensor,
    data_scaled=data_scaled,
    predicted_states=predicted_states,
    key_features=key_features,
    original_df=df,
    window_size=5
)


# =============================
# Step 6: Optional Insights
# =============================

print("\n" + "="*60)
print("OPTIONAL INSIGHTS & NEXT STEPS")
print("="*60)

# Interpret HMM states by average feature values
print("\nAverage Feature Values per HMM State (for interpretation):")
state_means = df[key_features].groupby(predicted_states).mean()
print(state_means.round(3))

print(f"\n💡 Tip: Map HMM states to maneuvers (e.g., State 0 = Ground, State 2 = Climb) based on above means.")

print("\n💾 Model Persistence:")
print("torch.save(model.state_dict(), 'results-AeroML/autoencoder.pth')")
print("import pickle; pickle.dump(hmm_model, open('results-AeroML/hmm_model.pkl', 'wb'))")

print("\n📈 Extensions:")
print("- Tune window_size, n_components, or embed_dim via grid search.")
print("- Add confidence intervals to HMM forecasts.")
print("- Use DTW or shapelets for finer maneuver alignment.")
print("- Export to KML for Google Earth visualization.")