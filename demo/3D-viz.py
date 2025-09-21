import pandas as pd
import plotly.graph_objects as go

# Load CSV file
file_path = "/Users/adityagirish/capstoned/final_synthetic_phugoid_flight.csv"   # <-- Change this to your file path
df = pd.read_csv(file_path)

# Extract required columns
lat = df["latitude"]
lon = df["longitude"]
alt = df["altmsl"]

# Create 3D scatter plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=lon, y=lat, z=alt,
        mode="lines+markers",
        marker=dict(size=3, color=alt, colorscale="Viridis"),
        line=dict(color="blue", width=2),
        name="Flight Path"
    ),
    go.Scatter3d(
        x=[lon.iloc[0]], y=[lat.iloc[0]], z=[alt.iloc[0]],
        mode="markers+text",
        marker=dict(size=6, color="green"),
        text=["Start"],
        textposition="top center",
        name="Start"
    ),
    go.Scatter3d(
        x=[lon.iloc[-1]], y=[lat.iloc[-1]], z=[alt.iloc[-1]],
        mode="markers+text",
        marker=dict(size=6, color="red"),
        text=["Stop"],
        textposition="top center",
        name="Stop"
    )
])

# Set layout
fig.update_layout(
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title="Altitude (MSL)"
    ),
    title="3D Flight Path",
    showlegend=True
)

# Show the plot in browser
fig.show()
