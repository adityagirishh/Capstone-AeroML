#!/bin/bash

# Flight Black Box Anomaly Detection - Setup Script
echo "Setting up Flight Black Box Anomaly Detection Environment..."

# Create project directory
mkdir -p flight_anomaly_detection
cd flight_anomaly_detection

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy pandas scikit-learn matplotlib seaborn
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install plotly jupyter
pip install scipy statsmodels
pip install tsfresh pyod
pip install rpy2  # For R integration if needed

# Clone repositories
echo "Cloning repositories..."

# TranAD - Transformer-based anomaly detection
git clone https://github.com/imperial-qore/TranAD.git
cd TranAD
pip install -r requirements.txt
cd ..

# LSTM AutoEncoder
git clone https://github.com/vincrichard/LSTM-AutoEncoder-Unsupervised-Anomaly-Detection.git

# Warehouse Anomaly (ConvLSTM)
git clone https://github.com/alexisbdr/warehouse-anomaly.git

# ISODATA Clustering
git clone https://github.com/xavi-rp/ISODATA_clustering.git

# Create data directory
mkdir -p data
mkdir -p results
mkdir -p models

# Install additional dependencies for specific algorithms
echo "Installing additional dependencies..."
pip install tslearn
pip install stumpy  # For matrix profile
pip install ruptures  # For change point detection
pip install sktime
pip install pykalman
pip install hdbscan

# Create requirements file
echo "Creating requirements.txt..."
pip freeze > requirements.txt

echo "Setup complete! Please place your CSV file in the data/ directory."
echo "Activate the environment with: source venv/bin/activate"
echo "Run the main program with: python flight_anomaly_detector.py"