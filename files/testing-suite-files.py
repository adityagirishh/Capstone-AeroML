import os
import csv
from pathlib import Path

def count_csv_rows(file_path):
    """Count the number of rows in a CSV file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)
        return row_count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def parse_csv_directory(directory_path):
    """Parse a directory and list all CSV files with their row counts."""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    if not directory.is_dir():
        print(f"'{directory_path}' is not a directory.")
        return
    
    # Find all CSV files in the directory
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{directory_path}'.")
        return
    
    print(f"CSV files in '{directory_path}':")
    print("-" * 50)
    
    total_rows = 0
    res = []
    for csv_file in csv_files:
        row_count = count_csv_rows(csv_file)
        if row_count is not None:
            print(f"{csv_file.name}: {row_count} rows")
            total_rows += row_count
            res.append(row_count)
    
    
    print("-" * 50)
    print("maximum number of rows: ", max(res))
    print(f"Total files: {len(csv_files)}")
    print(f"Total rows across all files: {total_rows}")

'''
(base) adityagirish@MacBookPro capstoned % /opt/anaconda3/bin/python /Users/adityagirish/capstoned/existing_methods_te
sting.py
CSV files in '/Users/adityagirish/Desktop/capstoned/dataset copy':
--------------------------------------------------
rud_resp_L_long.csv: 2002 rows
stall_no_flap.csv: 4802 rows
phugoid_full_flap_stall_entry.csv: 5602 rows
spin_no_flap_L_high_rate.csv: 6402 rows
spin_full_flap_L_high_rate.csv: 4002 rows
elev_resp_3U_round.csv: 1202 rows
rud_resp_R_long.csv: 3602 rows
deep_stall.csv: 5202 rows
spin_half_flap_L_high_rate.csv: 5202 rows
ail_resp_R.csv: 1502 rows
ail_resp_2L-2R.csv: 4002 rows
idle_descent_trim-2_52deg.csv: 2802 rows
ail_resp_L-R.csv: 2402 rows
landing_half_flap.csv: 5922 rows
rud_resp_L_R_short.csv: 4002 rows
stall_full_flap_high_rate.csv: 2402 rows
split_S.csv: 2402 rows
phugoid_no_flap_high_rate.csv: 7602 rows
phugoid_full_flap_push_entry.csv: 14802 rows
roll.csv: 3602 rows
elev_resp_3D_round.csv: 1202 rows
spin_no_flap_R_high_rate.csv: 7602 rows
stall_half_flap_low_rate.csv: 6002 rows
rud_resp_2R_L_short.csv: 3202 rows
idle_descent_trim-1_87deg.csv: 3602 rows
elev_resp_U_D_high_rate.csv: 2402 rows
stall_no_flap_low_rate.csv: 9602 rows
idle_descent_trim-2_18deg_curve.csv: 3202 rows
spin_full_flap_R_high_rate.csv: 6402 rows
idle_descent_half_flap.csv: 4802 rows
spin_half_flap_R_high_rate.csv: 4802 rows
stall_half_flap_high_rate.csv: 6802 rows
stall_no_flap_high_rate.csv: 3202 rows
rud_resp_R_short.csv: 1402 rows
idle_descent_trim-2_18deg.csv: 6002 rows
elev_resp_3U_square.csv: 2802 rows
rud_resp_R_L_short.csv: 2402 rows
ail_resp_R-L.csv: 2002 rows
rud_resp_L_short.csv: 1602 rows
ail_resp_L.csv: 1402 rows
--------------------------------------------------
maximum number of rows:  14802 -> phugoid_full_flap_push_entry.csv: 14802 rows
files to test on:
phugoid_full_flap_push_entry.csv : 14802
stall_no_flap_low_rate.csv: 9602 rows
spin_no_flap_R_high_rate.csv: 7602 rows
stall_half_flap_high_rate.csv: 6802 rows
idle_descent_trim-2_18deg.csv: 6002 rows
Total files: 40
Total rows across all files: 169900
'''

print("since this is the maximum number of rows, we will use it for testing of the existing methods")

if __name__ == "__main__":
    directory_path = "/Users/adityagirish/Desktop/capstoned/dataset copy"
    parse_csv_directory(directory_path)