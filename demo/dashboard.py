import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from PIL import Image
import io
import google.generativeai as genai

# Configure Gemini API - replace with your actual API key
genai.configure(api_key="AIzaSyBs1qzPOo8JwJqxAI9B2xT3mshZY4T5IBs")  # Add your API key

# Prompt-tuned setup for Gemini
FLIGHT_PROMPT_TEMPLATE = """
You are an expert in flight maneuvers, trained only on flight training data. Analyze the following flight data segment for cluster {cluster_id}.
Data summary: {data_summary}
Image description: The image shows a 3D flight path with colored clusters. Cluster {cluster_id} is colored {color} and represents a maneuver like {maneuver_example}.
Provide explainable insights including:
- Maneuver type (e.g., PHUGOID)
- Confidence score (0-1)
- Key features contributing to this maneuver
- Correlation with color in the image
Do not use external knowledge beyond flight training data.
"""

# Color mapping based on the image legend
COLOR_MAP = {
    0: "Blue",
    1: "Orange",
    2: "Green",
    3: "Red",
    4: "Purple",
    5: "Brown"
}

# Maneuver examples
MANEUVER_EXAMPLES = {
    2: "PHUGOID (blue squiggly line)"
}

def get_llm_insights(cluster_id, cluster_data, image):
    data_summary = cluster_data.describe().to_string()
    color = COLOR_MAP.get(cluster_id, "Unknown")
    maneuver_example = MANEUVER_EXAMPLES.get(cluster_id, "unknown maneuver")
    
    prompt = FLIGHT_PROMPT_TEMPLATE.format(
        cluster_id=cluster_id,
        data_summary=data_summary,
        color=color,
        maneuver_example=maneuver_example
    )
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([
        prompt,
        {"mime_type": "image/png", "data": img_byte_arr}
    ])
    
    return response.text

# Streamlit app
st.title("Flight Maneuver Explainability Dashboard")

# File uploads
csv_file = st.file_uploader("Upload CSV file (segmented_flight_data_v2-2.csv)", type="csv")
image_file = st.file_uploader("Upload Image (flight path plot)", type=["png", "jpg", "jpeg"])

if csv_file and image_file:
    # Load data
    df = pd.read_csv(csv_file)
    image = Image.open(image_file)
    
    st.image(image, caption="Uploaded Flight Path Image", use_column_width=True)
    
    # Select features
    feature_cols = ['latitude', 'longitude', 'altmsl', 'ias', 'gndspd', 'vspd', 'pitch', 'roll', 
                    'latac', 'normac', 'hdg', 'trk', 'volt1', 'volt2', 'amp1', 'amp2', 
                    'fqtyl', 'fqtyr', 'e1 fflow', 'e1 oilt', 'e1 oilp', 'e1 rpm', 
                    'e1 egt4', 'tas', 'wnddr', 'pitch_rate', 'roll_rate', 'alt_rate', 
                    'speed_accel', 'enu_e', 'enu_n']
    X = df[[col for col in feature_cols if col in df.columns]].fillna(0)
    available_features = list(X.columns)
    y = df['hmm_state']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train surrogate model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # LIME explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=available_features,
        class_names=[f"Cluster {i}" for i in np.unique(y)],
        mode="classification"
    )
    
    # Group by clusters
    clusters = df['hmm_state'].unique()
    clusters.sort()
    
    for cluster_id in clusters:
        st.header(f"Cluster {cluster_id} Insights")
        
        cluster_data = df[df['hmm_state'] == cluster_id]
        if cluster_data.empty:
            st.write("No data for this cluster.")
            continue
        
        # Select a representative instance
        instance_idx = cluster_data.index[0]
        instance = X_scaled[instance_idx].reshape(1, -1)  # Ensure 2D array
        
        # LIME explanation
        st.subheader("LIME Explanation")
        exp_lime = explainer_lime.explain_instance(instance[0], model.predict_proba, num_features=5)
        fig_lime = exp_lime.as_pyplot_figure()
        st.pyplot(fig_lime)
        
        # LLM insights
        st.subheader("LLM-Based Insights (Gemini)")
        with st.spinner("Generating LLM insights..."):
            llm_output = get_llm_insights(cluster_id, cluster_data, image)
        st.write(llm_output)
        
        st.divider()
else:
    st.write("Please upload both the CSV file and the image to proceed.")