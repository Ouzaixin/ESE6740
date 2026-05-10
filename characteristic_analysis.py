import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal, chi2_contingency


# subtype_path = "./results/subtype_analysis/subtype_with_bestK.csv"
subtype_path ="./results/subtype_analysis/AD_only/AD_only_subtype_with_bestK.csv"
char_path = "./dataset/characteristic.csv"
save_dir = "./results/subtype_analysis"
os.makedirs(save_dir, exist_ok=True)


# ------------------------------------------------------------
# Load and merge
# ------------------------------------------------------------
sub_df = pd.read_csv(subtype_path)
char_df = pd.read_csv(char_path)

sub_df["PTID"] = sub_df["PTID"].astype(str)
char_df["PTID"] = char_df["PTID"].astype(str)

# Keep only characteristic columns needed for analysis.
char_cols = [
    "PTID",
    "DX_bl",
    "AGE",
    "PTEDUCAT",
    "PTGENDER",
    "APOE4",
]

char_cols = [c for c in char_cols if c in char_df.columns]
char_df = char_df[char_cols].copy()

df = pd.merge(sub_df, char_df, on="PTID", how="inner", suffixes=("_sub", ""))

# Required fields
required_cols = [
    "subtype",
    "DX_bl",
    "AGE",
    "PTEDUCAT",
    "PTGENDER",
    "APOE4",
]

df = df.dropna(subset=required_cols).copy()

df["subtype"] = df["subtype"].astype(int)
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["PTEDUCAT"] = pd.to_numeric(df["PTEDUCAT"], errors="coerce")
df["PTGENDER"] = df["PTGENDER"].astype(str)
df["APOE4"] = pd.to_numeric(df["APOE4"], errors="coerce").astype(int).astype(str)
df["DX_bl"] = df["DX_bl"].astype(str)

df = df.dropna(subset=["AGE", "PTEDUCAT", "APOE4"]).copy()

print("Subjects after merge/dropna:", len(df))
print("Subtypes:", sorted(df["subtype"].unique()))


# ------------------------------------------------------------
# Continuous summaries
# ------------------------------------------------------------
cont_table = df.groupby("subtype")[["AGE", "PTEDUCAT"]].agg(["mean", "std", "median", "count"])
cont_table.to_csv(os.path.join(save_dir, "continuous_summary.csv"))

print("\n=== Continuous Summary ===")
print(cont_table)


# ------------------------------------------------------------
# Categorical summaries
# ------------------------------------------------------------
def save_crosstab(var, prefix):
    cnt = pd.crosstab(df["subtype"], df[var])
    pct = pd.crosstab(df["subtype"], df[var], normalize="index") * 100

    cnt.to_csv(os.path.join(save_dir, f"{prefix}_count.csv"))
    pct.to_csv(os.path.join(save_dir, f"{prefix}_percent.csv"))

    print(f"\n=== {prefix} count ===")
    print(cnt)

    print(f"\n=== {prefix} percent ===")
    print(pct)

    return cnt, pct


dx_cnt, dx_pct = save_crosstab("DX_bl", "diagnosis")
gender_cnt, gender_pct = save_crosstab("PTGENDER", "gender")
apoe_cnt, apoe_pct = save_crosstab("APOE4", "apoe")


# ------------------------------------------------------------
# Statistical tests
# ------------------------------------------------------------
stats_rows = []

# Kruskal-Wallis for continuous variables
for var in ["AGE", "PTEDUCAT"]:
    groups = [
        g[var].dropna().values
        for _, g in df.groupby("subtype")
        if len(g[var].dropna()) > 0
    ]

    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        stats_rows.append({
            "variable": var,
            "test": "Kruskal-Wallis",
            "statistic": stat,
            "p_value": p,
        })

# Chi-square for categorical variables
for var in ["DX_bl", "PTGENDER", "APOE4"]:
    table = pd.crosstab(df["subtype"], df[var])

    if table.shape[0] >= 2 and table.shape[1] >= 2:
        stat, p, dof, expected = chi2_contingency(table)
        stats_rows.append({
            "variable": var,
            "test": "Chi-square",
            "statistic": stat,
            "p_value": p,
            "dof": dof,
        })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(save_dir, "subtype_characteristic_tests.csv"), index=False)

print("\n=== Statistical Tests ===")
print(stats_df)


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
sns.set(style="whitegrid")


# AGE
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="subtype", y="AGE")
sns.stripplot(data=df, x="subtype", y="AGE", color="black", alpha=0.25, size=2)
plt.title("Age by subtype")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "age_boxplot.png"), dpi=150)
plt.close()


# EDUCATION
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="subtype", y="PTEDUCAT")
sns.stripplot(data=df, x="subtype", y="PTEDUCAT", color="black", alpha=0.25, size=2)
plt.title("Education by subtype")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "education_boxplot.png"), dpi=150)
plt.close()


def plot_percent_bar(pct_table, var_name, title, filename):
    plot_df = pct_table.reset_index().melt(
        id_vars="subtype",
        var_name=var_name,
        value_name="Percent",
    )

    plt.figure(figsize=(7, 4))
    sns.barplot(data=plot_df, x="subtype", y="Percent", hue=var_name)
    plt.title(title)
    plt.ylabel("Percent within subtype")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()


plot_percent_bar(
    dx_pct,
    var_name="Diagnosis",
    title="Diagnosis distribution by subtype (%)",
    filename="diagnosis_barplot.png",
)

plot_percent_bar(
    gender_pct,
    var_name="Gender",
    title="Gender distribution by subtype (%)",
    filename="gender_barplot.png",
)

plot_percent_bar(
    apoe_pct,
    var_name="APOE4",
    title="APOE4 distribution by subtype (%)",
    filename="apoe_barplot.png",
)


# ------------------------------------------------------------
# Save merged analysis dataframe
# ------------------------------------------------------------
df.to_csv(os.path.join(save_dir, "subtype_characteristics_merged.csv"), index=False)

print("\nDone. Results saved to:", save_dir)