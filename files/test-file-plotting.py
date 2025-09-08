import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

def load_flight_data(csv_file_path):
    """Load CSV file and validate required columns."""
    try:
        df = pd.read_csv(csv_file_path)
        
        # Common column name variations
        lat_cols = ['Lat', 'Latitude', 'lat', 'latitude', 'LAT']
        lon_cols = ['Long', 'Longitude', 'lon', 'longitude', 'LONG', 'Lon']
        alt_cols = ['Alt', 'Altitude', 'alt', 'altitude', 'ALT']
        
        # Find the actual column names
        lat_col = next((col for col in lat_cols if col in df.columns), None)
        lon_col = next((col for col in lon_cols if col in df.columns), None)
        alt_col = next((col for col in alt_cols if col in df.columns), None)
        
        if not all([lat_col, lon_col, alt_col]):
            print(f"Required columns not found in {Path(csv_file_path).name}")
            print("Available columns:", list(df.columns))
            print("Looking for: Latitude, Longitude, Altitude (or similar)")
            return None, None, None, None
        
        print(f"Found columns in {Path(csv_file_path).name}: {lat_col}, {lon_col}, {alt_col}")
        return df, lat_col, lon_col, alt_col
        
    except Exception as e:
        print(f"Error loading {Path(csv_file_path).name}: {e}")
        return None, None, None, None

def create_3d_flight_path(df, lat_col, lon_col, alt_col, title="3D Flight Path"):
    """Create a 3D flight path visualization."""
    
    # Remove rows with missing data
    df_clean = df.dropna(subset=[lat_col, lon_col, alt_col])
    
    if df_clean.empty:
        print("No valid data points found after removing NaN values.")
        return None
    
    print(f"Plotting {len(df_clean)} data points...")
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    # Add the flight path as a line
    fig.add_trace(go.Scatter3d(
        x=df_clean[lon_col],
        y=df_clean[lat_col],
        z=df_clean[alt_col],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=df_clean[alt_col],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Altitude")
        ),
        line=dict(
            color='red',
            width=2
        ),
        name='Flight Path',
        hovertemplate=
        '<b>Latitude:</b> %{y:.6f}<br>' +
        '<b>Longitude:</b> %{x:.6f}<br>' +
        '<b>Altitude:</b> %{z:.2f}<br>' +
        '<extra></extra>'
    ))
    
    # Add start point
    fig.add_trace(go.Scatter3d(
        x=[df_clean[lon_col].iloc[0]],
        y=[df_clean[lat_col].iloc[0]],
        z=[df_clean[alt_col].iloc[0]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Start',
        hovertemplate='<b>START</b><extra></extra>'
    ))
    
    # Add end point
    fig.add_trace(go.Scatter3d(
        x=[df_clean[lon_col].iloc[-1]],
        y=[df_clean[lat_col].iloc[-1]],
        z=[df_clean[alt_col].iloc[-1]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='End',
        hovertemplate='<b>END</b><extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700
    )
    
    return fig

def process_flight_files():
    """Process all specified flight data files and create visualizations."""
    
    # Define the base directory and output directory
    base_dir = "/Users/adityagirish/Desktop/capstoned/dataset copy"
    output_dir = "testing-dataset-plots"
    
    # List of files to process
    filenames = [
        "phugoid_full_flap_push_entry.csv",
        "stall_no_flap_low_rate.csv", 
        "spin_no_flap_R_high_rate.csv",
        "stall_half_flap_high_rate.csv",
        "idle_descent_trim-2_18deg.csv"
    ]
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Created/Using output directory: {output_dir}")
    print("=" * 60)
    
    successful_plots = 0
    failed_plots = 0
    
    for filename in filenames:
        print(f"\nProcessing: {filename}")
        print("-" * 40)
        
        # Construct full file path
        csv_file_path = Path(base_dir) / filename
        
        if not csv_file_path.exists():
            print(f"❌ File not found: {csv_file_path}")
            failed_plots += 1
            continue
        
        # Load the data
        df, lat_col, lon_col, alt_col = load_flight_data(csv_file_path)
        
        if df is None:
            failed_plots += 1
            continue
        
        print(f"📊 Loaded {len(df)} rows of data")
        print(f"🔢 Altitude range: {df[alt_col].min():.2f} to {df[alt_col].max():.2f}")
        print(f"🌍 Latitude range: {df[lat_col].min():.6f} to {df[lat_col].max():.6f}")
        print(f"🌍 Longitude range: {df[lon_col].min():.6f} to {df[lon_col].max():.6f}")
        
        # Create the visualization
        file_stem = csv_file_path.stem
        fig = create_3d_flight_path(df, lat_col, lon_col, alt_col, 
                                   title=f"3D Flight Path - {file_stem.replace('_', ' ').title()}")
        
        if fig is None:
            failed_plots += 1
            continue
        
        # Save the plot
        output_file = Path(output_dir) / f"{file_stem}_3d_plot.html"
        fig.write_html(output_file)
        print(f"✅ Plot saved: {output_file}")
        
        # Also save a PNG version (requires kaleido: pip install kaleido)
        try:
            png_file = Path(output_dir) / f"{file_stem}_3d_plot.png"
            fig.write_image(png_file, width=1200, height=800)
            print(f"📷 PNG saved: {png_file}")
        except Exception as e:
            print(f"⚠️  Could not save PNG (install kaleido for PNG export): {e}")
        
        successful_plots += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {successful_plots} files")
    print(f"❌ Failed to process: {failed_plots} files")
    print(f"📁 All plots saved in: {Path(output_dir).absolute()}")
    
    if successful_plots > 0:
        print(f"\n🎉 You can now open the HTML files in your browser to view the interactive 3D plots!")

if __name__ == "__main__":
    print("🚁 Batch Flight Path Visualizer")
    print("Processing specified flight data files...")
    print("=" * 60)
    
    process_flight_files()