import pandas as pd
file_path= "/Users/adityagirish/capstoned/data-discrepancy-testing/final_log_210115_104702_VAJB.csv"
df = pd.read_csv(file_path)

print(df.describe())