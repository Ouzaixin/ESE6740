import os
import pandas as pd

# input_csv = "./dataset/UCBERKELEY_AMY_6MM_24Mar2026.csv"
input_csv = "./dataset/ADNIMERGE_31Mar2026.csv"
output_csv = "./dataset/characteristic.csv"

PT_ID_COL = "PTID"
# DATE_COL = "SCANDATE"
DATE_COL = "EXAMDATE"

wanted_cols = [
    "PTID",
    # "SCANDATE",
    "EXAMDATE",
    "DX_bl",
    "AGE",
    "PTEDUCAT",
    "PTGENDER",
    "APOE4",
]

df = pd.read_csv(input_csv)

# Check which requested columns exist
available_cols = [c for c in wanted_cols if c in df.columns]
missing_cols = [c for c in wanted_cols if c not in df.columns]

print("Available columns:", available_cols)
print("Missing columns:", missing_cols)

# Keep available characteristic columns
char_df = df[available_cols].copy()

# Use earliest scan per subject, matching your preprocessing logic
if DATE_COL in char_df.columns:
    char_df[DATE_COL] = pd.to_datetime(char_df[DATE_COL], errors="coerce")
    char_df = char_df.sort_values([PT_ID_COL, DATE_COL])
else:
    char_df = char_df.sort_values([PT_ID_COL])

char_df = char_df.groupby(PT_ID_COL, as_index=False).first()

# Ensure PTID is string for merging
char_df[PT_ID_COL] = char_df[PT_ID_COL].astype(str)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
char_df.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print("Subjects:", len(char_df))
print("Columns:", list(char_df.columns))