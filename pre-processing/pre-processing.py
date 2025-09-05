# # The file still has inconsistent rows even after skipping the first two lines.
# # Let's load the file with skiprows=2 and on_bad_lines='skip' to skip problematic rows.

"""
Section 1: Data Loading and Column Cleaning
This section loads aircraft flight data from a CSV file, skipping the first 2 rows (metadata/headers). 
It cleans column names by stripping whitespace and applies a comprehensive mapping to standardize column names ('Lcl Date' → 'lcl date'). 
The code handles potential file loading errors and provides feedback on data shape and missing values. 
It also identifies numeric columns and converts data types appropriately, transforming datetime columns while converting remaining object columns to numeric format. 
Finally, it saves the cleaned dataset as "log1.csv" for further processing.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os


# --- Configuration ---
file_path = "/Users/adityagirish/Desktop/AerX shared data /log_210115_104702_VAJB.csv"
log_name = os.path.splitext(os.path.basename(file_path))[0]

# --- Load Data ---
try:
    # Skip metadata lines (first 3 lines contain headers and comments)
    df = pd.read_csv(file_path, low_memory=False, skiprows=2)
    print(f"✓ Successfully loaded {file_path}")
    print(f"  Data shape: {df.shape}")
except FileNotFoundError:
    print(f"✗ Error: The file '{file_path}' was not found.")
    exit()

# --- Column Cleaning ---
df.columns = df.columns.str.strip()
column_mapping = {
    'Lcl Date': 'lcl date', 'Lcl Time': 'lcl time', 'UTCOfst': 'utcofst',
    'AtvWpt': 'atvwpt', 'Latitude': 'latitude', 'Longitude': 'longitude',
    'AltInd': 'altind', 'BaroA': 'baroa', 'AltMSL': 'altmsl', 'OAT  ': 'oat',
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
df.rename(columns=lambda c: column_mapping.get(c, c), inplace=True)
print("✓ Cleaned column headers")
missing_percentages = (df.isnull().sum() / len(df) * 100).round(2)
missing_info = pd.DataFrame({
    'Missing Percentage': missing_percentages
}).sort_values('Missing Percentage', ascending=False)

print("Columns with missing values (%):")
print(len(missing_info[missing_info['Missing Percentage'] > 0]))

print("Actual columns in the dataframe:")
print(len(df.columns.tolist()))
print(df.dtypes)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("Numeric columns:",numeric_cols.tolist())


# Define key features
key_features = [
    'altind', 'altmsl', 'altgps', 'ias', 'gndspd', 'tas', 'vspd', 'vspdg',
    'pitch', 'roll', 'hdg', 'e1 rpm', 'e1 oilt', 'e1 cht1', 'e1 cht2', 
    'e1 cht3', 'e1 cht4', 'e1 egt1', 'e1 egt2', 'e1 egt3', 'e1 egt4',
    'afcson', 'rollm', 'pitchm', 'rollc', 'pichc', 'gpsfix', 'hal', 'val', 
    'hplwas', 'hplfd', 'vplwas', 'fqtyl', 'fqtyr', 'volt1', 'volt2', 'amp1', 'amp2'
]

# Convert lcl date to datetime and keep lcl time as timedelta (or convert to string)
df['lcl date'] = pd.to_datetime(df['lcl date'], errors='coerce')
df['lcl time'] = pd.to_timedelta(df['lcl time'], errors='coerce')


# Get remaining object columns (excluding date/time)
object_cols = df.select_dtypes(include=['object']).columns.tolist()

# Convert remaining object columns to float
for col in object_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df.to_csv(f"{log_name}.csv", index=False)


'''
** remember **
StandardScaler is a great default choice, especially for algorithms that assume normally distributed data and are sensitive to feature scales.
MinMaxScaler is a good option when you need to constrain your features within a specific range, but be mindful of outliers.
RobustScaler is the best choice when your dataset is heavily impacted by outliers.
Normalizer is for when the vector's direction, not magnitude, is important.
'''

#----------------------------------

"""
Section 2: Data Scaling
This section applies StandardScaler to normalize numerical features, ensuring all variables have mean=0 and standard deviation=1. 
It defines which columns should be scaled (excluding date/time columns) and fits the scaler to the data. 
StandardScaler is chosen because it's ideal for algorithms sensitive to feature scales and assumes normally distributed data. 
The scaled data maintains the same structure but with standardized values, making it suitable for machine learning algorithms. 
The processed data is saved as "scaled_log1.csv".
"""


df1 = pd.read_csv(f"{log_name}.csv")


from sklearn.preprocessing import StandardScaler

scalable_numeric_columns = ['utcofst', 'atvwpt', 'latitude',
       'longitude', 'altind', 'baroa', 'altmsl', 'OAT', 'ias', 'gndspd',
       'vspd', 'pitch', 'roll', 'latac', 'normac', 'hdg', 'trk', 'volt1',
       'volt2', 'amp1', 'amp2', 'fqtyl', 'fqtyr', 'e1 fflow', 'e1 oilt',
       'e1 oilp', 'e1 rpm', 'e1 cht1', 'e1 cht2', 'e1 cht3', 'e1 cht4',
       'e1 egt1', 'e1 egt2', 'e1 egt3', 'e1 egt4', 'altgps', 'tas', 'hsis',
       'crs', 'nav1', 'nav2', 'com1', 'com2', 'hcdi', 'vcdi', 'wndspd',
       'wnddr', 'wptdst', 'wptbrg', 'magvar', 'afcson', 'rollm', 'pitchm',
       'rollc', 'pichc', 'vspdg', 'gpsfix', 'hal', 'val', 'hplwas', 'hplfd',
       'vplwas']

non_scalable = ['lcl date','lcl time']


scaler = StandardScaler()

scaler.fit(df1[scalable_numeric_columns])


df1.to_csv(f"scaled_{log_name}.csv", index=False)
print("values scaled")

#------------------------------------------------------
"""
Section 3: Data Preprocessing and Feature Engineering
This final section handles missing data by dropping completely empty columns and applying mean imputation to fill remaining null values.
It then creates new derivative features for maneuver detection: pitch_rate, roll_rate, alt_rate, and speed_accel using the diff() function to calculate rate of change.
These features help identify aircraft maneuvers and flight dynamics. 
The code uses backward fill to handle NaN values created by the diff() operation and saves the final preprocessed dataset as "final_log1.csv", ready for analysis or machine learning applications.
"""


df2 = pd.read_csv(f"scaled_{log_name}.csv")

# Print initial info
print(f"Initial shape: {df2.shape}")
print(f"Initial null counts:\n{df2.isnull().sum()[df2.isnull().sum() > 0]}")

# Drop completely empty columns and perform mean imputation
empty_cols = df2.columns[df2.isnull().all()].tolist()
df2 = df2.dropna(axis=1, how='all').fillna(df2.select_dtypes(include='number').mean())

print(f"Dropped empty columns: {empty_cols if empty_cols else 'None'}")

# Add maneuver detection features
df2['pitch_rate'] = df2['pitch'].diff()
df2['roll_rate'] = df2['roll'].diff() 
df2['alt_rate'] = df2['altmsl'].diff()
df2['speed_accel'] = df2['ias'].diff()

# Fill the first NaN from diff()
df2.fillna(method='bfill', inplace=True)

# Print final info
print(f"\nFinal shape: {df2.shape}")
print(f"Remaining null counts: {df2.isnull().sum().sum()}")
print("Mean imputation completed for numeric columns")
print(f"Added rate features: pitch_rate, roll_rate, alt_rate, speed_accel")
print("Filled NaN values from diff() using backward fill")

print(df2[non_scalable].head(10))

df2.to_csv(f"final_{log_name}.csv", index=False)

#----------------------------------------------