import os
import pandas as pd

curr_dir = os.getcwd()
# print(curr_dir)
input_csv = f"{curr_dir}/dataset/UCBERKELEY_AMY_6MM_24Mar2026.csv"
output_dir = f"{curr_dir}/dataset/"
os.makedirs(output_dir, exist_ok=True)

PT_ID_COL = "PTID"
DATE_COL = "SCANDATE"

df = pd.read_csv(input_csv)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

mri_cols = [c for c in df.columns if "_VOLUME" in c.upper()]
pet_cols = [c for c in df.columns if "_SUVR" in c.upper()]

print(f"MRI ROIs: {len(mri_cols)}")
print(f"PET ROIs: {len(pet_cols)}")

def base_name(col, suffix):
    return col.upper().replace(suffix, "")

mri_map = {base_name(c, "_VOLUME"): c for c in mri_cols}
pet_map = {base_name(c, "_SUVR"): c for c in pet_cols}

common_rois = sorted(set(mri_map.keys()).intersection(set(pet_map.keys())))
print(f"Matched ROIs: {len(common_rois)}")

mri_final = [mri_map[r] for r in common_rois]
pet_final = [pet_map[r] for r in common_rois]

df = df.sort_values([PT_ID_COL, DATE_COL])
df = df.groupby(PT_ID_COL, as_index=False).first()

def filter_sparse(df, cols, threshold=0.7):
    valid_ratio = df[cols].notna().mean(axis=1)
    before = len(df)
    df = df[valid_ratio >= threshold].copy()
    after = len(df)
    print(f"Filtered sparse subjects: {before - after} removed, {after} kept")
    return df

mri_df = df[[PT_ID_COL, DATE_COL] + mri_final].copy()
pet_df = df[[PT_ID_COL, DATE_COL] + pet_final].copy()

mri_df = filter_sparse(mri_df, mri_final, threshold=0.7)
pet_df = filter_sparse(pet_df, pet_final, threshold=0.7)

common_ids = set(mri_df[PT_ID_COL]).intersection(set(pet_df[PT_ID_COL]))
print(f"Common subjects after filtering: {len(common_ids)}")

mri_df = mri_df[mri_df[PT_ID_COL].isin(common_ids)].copy()
pet_df = pet_df[pet_df[PT_ID_COL].isin(common_ids)].copy()

mri_df = mri_df.sort_values(PT_ID_COL).reset_index(drop=True)
pet_df = pet_df.sort_values(PT_ID_COL).reset_index(drop=True)

assert (mri_df[PT_ID_COL].values == pet_df[PT_ID_COL].values).all(), "TID mismatch after alignment!"

assert len(mri_final) == len(pet_final), "ROI count mismatch!"

assert set([base_name(c, "_VOLUME") for c in mri_final]) == \
       set([base_name(c, "_SUVR") for c in pet_final]), \
       "ROI semantic mismatch!"

mri_save = os.path.join(output_dir, "MRI_roi.csv")
pet_save = os.path.join(output_dir, "PET_roi.csv")

mri_df.to_csv(mri_save, index=False)
pet_df.to_csv(pet_save, index=False)

print("Saved MRI ROI:", mri_save)
print("Saved PET ROI:", pet_save)
print("Subjects:", len(mri_df))
print("ROI pairs:", len(common_rois))