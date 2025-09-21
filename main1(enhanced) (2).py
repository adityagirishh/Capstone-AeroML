"""
aero_segmentation_v2.py

Implements:
- lat/lon -> local ENU conversion
- derived features (deltas, rates, accelerations)
- multi-scale sliding windows
- LSTM autoencoder per-scale (shared weights optionally)
- concatenated multi-scale embeddings (fusion)
- clustering alternatives (HDBSCAN or GMM)
- HMM fit on embeddings (or cluster-posteriors -> HMM)
- HSMM fallback via min-duration smoothing if HSMM lib unavailable
- utilities: save/load models, basic eval (Boundary-F1, ARI)
"""

import os
import math
import pickle
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Optional libraries
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except Exception:
    HMMLEARN_AVAILABLE = False

# If pyhsmm or other HSMM libs installed, you could import; otherwise we use duration smoothing fallback
HSMM_AVAILABLE = False

# -------------------------
# CONFIG
# -------------------------
DATA_PATH =  "/Users/adityagirish/capstoned/final logs/final_log_210119_094547_VAJB.csv"  # change to your file
OUTPUT_DIR = "results-AeroML-v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature corpus (use your validated column names)
KEY_FEATURES_BASE = [
    'latitude', 'longitude', 'altmsl', 'altind', 'altgps', 'ias', 'gndspd', 'tas',
    'vspd', 'vspdg', 'pitch', 'roll', 'hdg', 'e1 rpm', 'e1 oilt', 'e1 cht1',
    'e1 cht2', 'e1 cht3', 'e1 cht4', 'e1 egt1', 'e1 egt2', 'e1 egt3', 'e1 egt4',
    'afcson', 'rollm', 'pitchm', 'rollc', 'pichc', 'gpsfix', 'hal', 'val',
    'hplwas', 'hplfd', 'vplwas', 'fqtyl', 'fqtyr', 'volt1', 'volt2', 'amp1', 'amp2'
]

SAMPLE_RATE_HZ = 1.0  # set to actual Hz if known
DT = 1.0 / SAMPLE_RATE_HZ

# Multi-scale window sizes in timesteps (expressed in seconds or timesteps depending on DT)
WINDOW_SIZES = [5, 15, 60]  # short, medium, long (adjust to data sample rate)
STEP = 1

# LSTM AE hyperparams
HIDDEN_DIM = 64
EMBED_DIM_PER_SCALE = 16   # each scale will produce this many dims; fused dim = sum(scales)
DROPOUT = 0.2
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100
PATIENCE = 10  # early stopping patience on val loss

# HMM / clustering params
N_HMM_STATES = 6  # initial guess (use model selection later)
USE_HDBSCAN = True if HDBSCAN_AVAILABLE else False
GMM_COMPONENTS = 6

# Duration smoothing (min duration in seconds / timesteps)
MIN_STATE_DURATION_SEC = 3.0
MIN_STATE_DURATION = int(max(1, round(MIN_STATE_DURATION_SEC * SAMPLE_RATE_HZ)))

# -------------------------
# UTILS: ENU conversion & derivatives
# -------------------------
def latlon_to_enu(lat, lon, ref_lat, ref_lon):
    """
    Approximate conversion to local ENU meters using small-angle approximations.
    Suitable for local flights (few tens of km).
    """
    # Reference point in radians
    R_earth = 6378137.0
    lat0 = math.radians(ref_lat)
    lon0 = math.radians(ref_lon)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    dlat = lat_rad - lat0
    dlon = lon_rad - lon0

    east = R_earth * dlon * np.cos(lat0)
    north = R_earth * dlat
    return east, north

def add_enu(df: pd.DataFrame, lat_col='latitude', lon_col='longitude'):
    ref_lat = float(df[lat_col].iloc[0])
    ref_lon = float(df[lon_col].iloc[0])
    east, north = latlon_to_enu(df[lat_col].values, df[lon_col].values, ref_lat, ref_lon)
    df = df.copy()
    df['enu_e'] = east
    df['enu_n'] = north
    return df

def add_time_derivatives(df: pd.DataFrame, cols: List[str], dt: float = DT, orders: int = 2):
    """
    Add derivative features up to 'orders' (e.g., 1 => rates, 2 => accelerations).
    Derived columns named as '{col}_d1', '{col}_d2', ...
    Uses simple finite differences; forward/backward fill edges.
    """
    df = df.copy()
    for col in cols:
        values = df[col].astype(float).values
        # first derivative
        d1 = np.gradient(values, dt)
        df[f"{col}_d1"] = d1
        if orders >= 2:
            d2 = np.gradient(d1, dt)
            df[f"{col}_d2"] = d2
    return df

# -------------------------
# Sliding windows & dataset helpers
# -------------------------
def create_windows_from_array(arr: np.ndarray, window_size: int, step: int = 1):
    n = len(arr)
    windows = []
    for i in range(0, n - window_size + 1, step):
        windows.append(arr[i:i+window_size])
    return np.array(windows)  # (n_windows, window_size, n_features)

def windows_to_center_timestep_indices(n_timesteps: int, window_size: int, step: int = 1):
    # return center indices for each window (for mapping window label -> timestep)
    centers = []
    for i in range(0, n_timesteps - window_size + 1, step):
        centers.append(i + window_size // 2)
    return np.array(centers)

def align_window_labels_to_timesteps(labels_windows: np.ndarray, window_size: int, total_timesteps: int):
    from scipy.stats import mode
    votes = [[] for _ in range(total_timesteps)]
    for win_start, label in enumerate(labels_windows):
        for offset in range(window_size):
            t = win_start + offset
            if t < total_timesteps:
                votes[t].append(int(label))
    aligned = np.full(total_timesteps, -1, dtype=int)
    for t, v in enumerate(votes):
        if len(v) > 0:
            aligned[t] = mode(v, keepdims=False).mode
    return aligned

def enforce_min_duration(sequence: np.ndarray, min_len: int):
    """
    Simple post-processing to enforce a minimum run length for state labels,
    by merging short runs with their neighbors (choose neighbor with longer run).
    """
    seq = sequence.copy()
    n = len(seq)
    i = 0
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        run_len = j - i
        if run_len < min_len:
            # choose neighbor to merge into: left or right based on run lengths
            left_len = 0
            right_len = 0
            # look left
            l = i - 1
            if l >= 0:
                val = seq[l]
                while l >= 0 and seq[l] == val:
                    left_len += 1
                    l -= 1
            # look right
            r = j
            if r < n:
                val = seq[r]
                while r < n and seq[r] == val:
                    right_len += 1
                    r += 1
            # merge into larger neighbor (prefer right if tie)
            if right_len >= left_len:
                seq[i:j] = seq[j] if j < n else seq[i-1] if i-1 >=0 else seq[i]
            else:
                seq[i:j] = seq[i-1] if i-1 >=0 else seq[j] if j < n else seq[i]
        i = j
    return seq

# -------------------------
# Model: LSTM Autoencoder (flexible per-scale)
# -------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, embed_dim:int, dropout:float=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bottleneck = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h, _) = self.encoder(x)
        h = h[-1]  # last layer hidden
        embed = self.bottleneck(h)
        embed = self.dropout(embed)
        embed_repeated = embed.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(embed_repeated)
        recon = self.output_layer(out)
        return recon

    def encode(self, x):
        with torch.no_grad():
            _, (h, _) = self.encoder(x)
            h = h[-1]
            embed = self.bottleneck(h)
        return embed

# -------------------------
# Training helper (with early stopping)
# -------------------------
def train_autoencoder(model, X_tensor, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE, device='cpu'):
    X = X_tensor.to(device)
    dataset = TensorDataset(X)
    # simple train/val split
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.85 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_loader = DataLoader(TensorDataset(X[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx]), batch_size=batch_size, shuffle=False)

    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inp = batch[0].to(device)
            recon = model(inp)
            loss = criterion(recon, inp)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * inp.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch[0].to(device)
                recon = model(inp)
                loss = criterion(recon, inp)
                val_loss += loss.item() * inp.size(0)
            val_loss /= max(1, len(val_loader.dataset))

        # print progress
        print(f"[AE] Epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("[AE] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -------------------------
# Main pipeline
# -------------------------
def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", DATA_PATH, "rows:", len(df))
    # select available features
    key_features = [c for c in KEY_FEATURES_BASE if c in df.columns]
    print("Using features:", key_features)

    # 1) ENU conversion
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = add_enu(df, 'latitude', 'longitude')
        enu_cols = ['enu_e', 'enu_n']
    else:
        enu_cols = []

    # 2) Derived features (rates, accelerations) for selected columns (including ENU, alt, ias, pitch, roll, heading)
    derive_cols = []
    for candidate in ['altmsl', 'ias', 'pitch', 'roll', 'hdg'] + enu_cols:
        if candidate in df.columns:
            derive_cols.append(candidate)
    df = add_time_derivatives(df, derive_cols, dt=DT, orders=2)

    # 3) Final feature list (base telemetry + derived)
    derived_generated = []
    for c in derive_cols:
        derived_generated += [f"{c}_d1", f"{c}_d2"]
    final_features = [c for c in key_features if c in df.columns] + enu_cols + derived_generated
    print("Final feature list count:", len(final_features))

    # 4) Missing values -> impute (interpolate then mean)
    df_ffill = df[final_features].interpolate(method='linear', limit_direction='both', axis=0)
    df_ffill = df_ffill.fillna(df_ffill.mean())
    X_raw = df_ffill.values.astype(float)  # shape (T, F)
    

    # 5) Standardize (fit scaler and persist)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 6) Multi-scale windows -> per-scale windows and autoencoder training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_embeddings = {}
    models_per_scale = {}
    centers_per_scale = {}
    for ws in WINDOW_SIZES:
        print(f"\n[Scale] window_size={ws} (timesteps)")
        windows = create_windows_from_array(X_scaled, window_size=ws, step=STEP)  # (n_windows, ws, F)
        if len(windows) == 0:
            print(f"[WARNING] Window size {ws} is larger than input length {len(X_scaled)}. Skipping this scale.")
            continue
        X_tensor = torch.tensor(windows, dtype=torch.float32)
        model = LSTMAutoencoder(input_dim=X_tensor.shape[2], hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM_PER_SCALE, dropout=DROPOUT)
        model = train_autoencoder(model, X_tensor, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE, device=device)
        # Extract embeddings (one per window)
        model.to(device)
        model.eval()
        with torch.no_grad():
            embeds = model.encode(X_tensor.to(device)).cpu().numpy()  # (n_windows, embed_dim)
        scale_embeddings[ws] = embeds
        models_per_scale[ws] = model
        centers_per_scale[ws] = windows_to_center_timestep_indices(len(X_scaled), ws, STEP)
        # save model
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"ae_ws{ws}.pth"))
        print(f"[Scale] embeddings shape: {embeds.shape}")

    # 7) Fusion: concatenate embeddings across scales matched by nearest center mapping
    # We will align per-window centers and create fused embedding for timesteps where all scales have windows (by window center)
    # Approach: build a dataframe keyed by center_timestep with embeddings from each scale
    fusion_df = {}
    for ws, embeds in scale_embeddings.items():
        centers = centers_per_scale[ws]
        for idx, c in enumerate(centers):
            if c not in fusion_df:
                fusion_df[c] = {}
            fusion_df[c][f"e_ws{ws}"] = embeds[idx]
    # Keep only centers that have embeddings from at least one scale (prefer all)
    fusion_keys = sorted(fusion_df.keys())
    fused_embeddings = []
    fused_centers = []
    for c in fusion_keys:
        parts = []
        for ws in WINDOW_SIZES:
            emb = fusion_df[c].get(f"e_ws{ws}")
            if emb is None:
                # pad with zeros if missing for simplicity
                parts.append(np.zeros(EMBED_DIM_PER_SCALE))
            else:
                parts.append(emb)
        fused = np.concatenate(parts)
        fused_embeddings.append(fused)
        fused_centers.append(c)
    fused_embeddings = np.array(fused_embeddings)  # (n_fused, sum(embed_dims))
    print("Fused embeddings shape:", fused_embeddings.shape)

    # Optional: reduce fused dim for speed/visualization
    pca = PCA(n_components=min(16, fused_embeddings.shape[1]))
    fused_embeddings_pca = pca.fit_transform(fused_embeddings)

    # 8) Clustering (HDBSCAN preferred if available)
    if USE_HDBSCAN:
        print("[Clustering] Using HDBSCAN")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=False)
        cluster_labels = clusterer.fit_predict(fused_embeddings)
        print("HDBSCAN clusters found:", len(np.unique(cluster_labels)))
    else:
        print("[Clustering] Using GMM")
        gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type='diag', random_state=42)
        gmm.fit(fused_embeddings)
        cluster_labels = gmm.predict(fused_embeddings)

    # Save raw fused embeddings + labels
    np.save(os.path.join(OUTPUT_DIR, "fused_embeddings.npy"), fused_embeddings)
    pd.DataFrame({
        "center_timestep": fused_centers,
        "cluster_label": cluster_labels
    }).to_csv(os.path.join(OUTPUT_DIR, "fused_cluster_labels.csv"), index=False)

    # 9) HMM on embeddings (fit HMM on fused embeddings or cluster posteriors)
    if HMMLEARN_AVAILABLE:
        # Use gaussian emissions on fused (or use PCA-reduced fused embeddings)
        hmm_model = hmm.GaussianHMM(n_components=N_HMM_STATES, covariance_type='diag', n_iter=200, random_state=42)
        print("[HMM] fitting to fused embeddings (PCA-reduced)")
        hmm_model.fit(fused_embeddings_pca)
        window_states = hmm_model.predict(fused_embeddings_pca)  # per fused-window/center
    else:
        print("[HMM] hmmlearn not available. Falling back to temporal smoothing of cluster labels.")
        # directly use cluster labels as window_states and proceed
        window_states = cluster_labels

    # Map window states back to full timesteps
    # fused_centers correspond to center timesteps => we will assign window state to the window-range (center +/- half window)
    # Simpler: align window labels to timesteps by expanding each fused center to contribute to a surrounding window of size = median WINDOW_SIZES
    median_ws = int(np.median(WINDOW_SIZES))
    half = median_ws // 2
    total_timesteps = len(X_scaled)
    timestep_votes = [[] for _ in range(total_timesteps)]
    for idx, center in enumerate(fused_centers):
        label = int(window_states[idx])
        start = max(0, center - half)
        end = min(total_timesteps, center + half + 1)
        for t in range(start, end):
            timestep_votes[t].append(label)
    # take mode or -1 if no votes
    from scipy.stats import mode
    aligned_states = np.full(total_timesteps, -1, dtype=int)
    for t in range(total_timesteps):
        if len(timestep_votes[t]) > 0:
            aligned_states[t] = mode(timestep_votes[t], keepdims=False).mode

    # 10) Duration modeling / HSMM or fallback smoothing
    if HSMM_AVAILABLE:
        # placeholder for HSMM usage - if library available implement here
        pass
    else:
        print("[Duration] Applying min-duration enforcement smoothing")
        aligned_states_smoothed = enforce_min_duration(aligned_states, MIN_STATE_DURATION)
    # save labels to CSV
    out_df = df.copy()
    out_df['hmm_state'] = aligned_states_smoothed
    out_df.to_csv(os.path.join(OUTPUT_DIR, "segmented_flight_data_v2.csv"), index=False)
    print("Saved segmented CSV:", os.path.join(OUTPUT_DIR, "segmented_flight_data_v2.csv"))

    # 11) Summary visuals
    plt.figure(figsize=(10,6))
    plt.plot(out_df['altmsl'], label='altmsl')
    plt.plot(out_df['hmm_state'] * (np.nanmax(out_df['altmsl']) - np.nanmin(out_df['altmsl'])) / (np.nanmax(out_df['hmm_state'])+1 + 1e-9), label='state (scaled)')
    plt.legend()
    plt.title("Altitude & State Overlay")
    plt.savefig(os.path.join(OUTPUT_DIR, "alt_state_overlay.png"), bbox_inches='tight')

    # 12) Save models & artifacts
    with open(os.path.join(OUTPUT_DIR, "fusion_pca.pkl"), "wb") as f:
        pickle.dump({"pca": pca, "windows": {"sizes": WINDOW_SIZES, "centers": fused_centers}}, f)
    if HMMLEARN_AVAILABLE:
        with open(os.path.join(OUTPUT_DIR, "hmm_model.pkl"), "wb") as f:
            pickle.dump(hmm_model, f)
    print("All artifacts saved to:", OUTPUT_DIR)

    # 13) Basic evaluation helpers (if ground-truth available)
    # Save boundary events for quick inspection
    boundaries = np.where(np.diff(out_df['hmm_state'], prepend=out_df['hmm_state'].iloc[0]) != 0)[0]
    pd.Series(boundaries).to_csv(os.path.join(OUTPUT_DIR, "detected_boundaries.csv"), index=False)
    print("Detected boundaries:", len(boundaries))

# -------------------------
# Basic evaluation: Boundary-F1 (within tolerance)
# -------------------------
def boundary_f1(pred_boundaries: List[int], true_boundaries: List[int], tol: int = 3):
    """
    pred_boundaries, true_boundaries: lists of timestep indices where boundaries occur
    tol: tolerance in timesteps
    """
    pred = np.array(pred_boundaries)
    true = np.array(true_boundaries)
    if len(pred) == 0 and len(true) == 0:
        return 1.0, 1.0, 1.0
    matched_pred = set()
    tp = 0
    for tb in true:
        # find any pred within +/- tol
        diffs = np.abs(pred - tb)
        close = np.where(diffs <= tol)[0]
        if close.size > 0:
            tp += 1
            matched_pred.add(close[0])
    fp = len(pred) - len(matched_pred)
    fn = len(true) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    main()
