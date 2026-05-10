import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

subtype_path = "./results/subtype_analysis/test_subtype_with_bestK.csv"
char_path = "./dataset/characteristic.csv"
save_dir = "./results/subtype_analysis"
os.makedirs(save_dir, exist_ok=True)

sub_df = pd.read_csv(subtype_path)
char_df = pd.read_csv(char_path)

sub_df["PTID"] = sub_df["PTID"].astype(str)
char_df["PTID"] = char_df["PTID"].astype(str)

df = pd.merge(sub_df, char_df, on="PTID", how="inner")

df = df.dropna(subset=[
    "subtype",
    "AGE",
    "PTEDUCAT",
    "PTGENDER",
    "APOE4",
    "DX_bl"
])


df["AGE"] = pd.to_numeric(df["AGE"])
df["PTEDUCAT"] = pd.to_numeric(df["PTEDUCAT"])
df["subtype"] = df["subtype"].astype(int)

df["PTGENDER"] = df["PTGENDER"].astype(str)
df["APOE4"] = df["APOE4"].astype(str)
df["DX_bl"] = df["DX_bl"].astype(str)

cont_table = df.groupby("subtype")[["AGE", "PTEDUCAT"]].agg(["mean", "std", "count"])
cont_table.to_csv(os.path.join(save_dir, "continuous_summary.csv"))

print("\n=== Continuous Summary ===")
print(cont_table)

gender_pct = pd.crosstab(df["subtype"], df["PTGENDER"], normalize="index") * 100
gender_cnt = pd.crosstab(df["subtype"], df["PTGENDER"])

gender_pct.to_csv(os.path.join(save_dir, "gender_percent.csv"))
gender_cnt.to_csv(os.path.join(save_dir, "gender_count.csv"))

print("\n=== Gender (%) ===")
print(gender_pct)

apoe_pct = pd.crosstab(df["subtype"], df["APOE4"], normalize="index") * 100
apoe_cnt = pd.crosstab(df["subtype"], df["APOE4"])

apoe_pct.to_csv(os.path.join(save_dir, "apoe_percent.csv"))
apoe_cnt.to_csv(os.path.join(save_dir, "apoe_count.csv"))

dlx_pct = pd.crosstab(df["subtype"], df["DX_bl"], normalize="index") * 100
dlx_cnt = pd.crosstab(df["subtype"], df["DX_bl"])

dlx_cnt.to_csv(os.path.join(save_dir, "dx_count.csv"))
dlx_pct.to_csv(os.path.join(save_dir, "dx_percent.csv"))

print("\n=== APOE4 (%) ===")
print(apoe_pct)

sns.set(style="whitegrid")

# AGE
plt.figure()
sns.boxplot(data=df, x="subtype", y="AGE")
plt.title("Age by subtype (NaN removed)")
plt.savefig(os.path.join(save_dir, "age_boxplot.png"), dpi=150)
plt.close()

# EDUCATION
plt.figure()
sns.boxplot(data=df, x="subtype", y="PTEDUCAT")
plt.title("Education by subtype (NaN removed)")
plt.savefig(os.path.join(save_dir, "education_boxplot.png"), dpi=150)
plt.close()

# GENDER
gender_plot = gender_pct.reset_index().melt(
    id_vars="subtype",
    var_name="Gender",
    value_name="Percent"
)

plt.figure()
sns.barplot(data=gender_plot, x="subtype", y="Percent", hue="Gender")
plt.title("Gender distribution (%)")
plt.savefig(os.path.join(save_dir, "gender_barplot.png"), dpi=150)
plt.close()

# APOE4
apoe_plot = apoe_pct.reset_index().melt(
    id_vars="subtype",
    var_name="APOE4",
    value_name="Percent"
)

plt.figure()
sns.barplot(data=apoe_plot, x="subtype", y="Percent", hue="APOE4")
plt.title("APOE4 distribution (%)")
plt.savefig(os.path.join(save_dir, "apoe_barplot.png"), dpi=150)
plt.close()

# DX
dx_plot = dlx_pct.reset_index().melt(
    id_vars="subtype",
    var_name="Diagnosis",
    value_name="Percent"
)
plt.figure()
sns.barplot(data=dx_plot, x="subtype", y="Percent", hue="Diagnosis")
plt.title("Diagnosis distribution (%)")
plt.savefig(os.path.join(save_dir, "dx_barplot.png"), dpi=150)
plt.close()

print("\nDone. All NaNs removed and results saved to:", save_dir)