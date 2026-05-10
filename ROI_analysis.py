import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, ttest_ind
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


MRI_PATH = "./dataset/PET_whole.csv"
CHAR_PATH = "./results/subtype_analysis/test_subtype_with_bestK.csv"
OUTPUT_DIR = "./results/subtype_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PTID_COL = "PTID"
SUBTYPE_COL = "subtype"  

mri_df = pd.read_csv(MRI_PATH)
char_df = pd.read_csv(CHAR_PATH)

print("MRI shape:", mri_df.shape)
print("CHAR shape:", char_df.shape)

mri_df[PTID_COL] = mri_df[PTID_COL].astype(str).str.strip()
char_df[PTID_COL] = char_df[PTID_COL].astype(str).str.strip()


print("Characteristic columns:", char_df.columns.tolist())

df = mri_df.merge(
    char_df[[PTID_COL, SUBTYPE_COL]],
    on=PTID_COL,
    how='inner'
)

print("After merge shape:", df.shape)

roi_cols = [
    c for c in df.columns
    if ('suvr' in c.lower())
]

print(f"Number of ROI features: {len(roi_cols)}")
print("Example ROI:", roi_cols[:10])

df = df.dropna(subset=roi_cols + [SUBTYPE_COL])

print("After dropna:", df.shape)

scaler = StandardScaler()
df[roi_cols] = scaler.fit_transform(df[roi_cols])

print("ROI standardized")

subtypes = sorted(df[SUBTYPE_COL].unique())
print("Subtypes:", subtypes)

roi_mean = df.groupby(SUBTYPE_COL)[roi_cols].mean()
roi_mean.to_csv(os.path.join(OUTPUT_DIR, "PET_roi_mean_by_subtype.csv"))

print("ROI mean saved")

anova_results = {}

for roi in roi_cols:
    groups = [df[df[SUBTYPE_COL] == k][roi] for k in subtypes]

    if any(len(g) == 0 for g in groups):
        continue

    stat, p = f_oneway(*groups)
    anova_results[roi] = p

anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['p_value'])
anova_df = anova_df.sort_values('p_value')


reject, pvals_corrected, _, _ = multipletests(anova_df['p_value'], method='fdr_bh')
anova_df['p_corrected'] = pvals_corrected
anova_df['significant'] = reject

anova_df.to_csv(os.path.join(OUTPUT_DIR, "PET_anova_results.csv"))
print("ANOVA done")

if len(subtypes) == 2:
    print("Running t-test")

    g0 = df[df[SUBTYPE_COL] == subtypes[0]]
    g1 = df[df[SUBTYPE_COL] == subtypes[1]]

    ttest_results = {}

    for roi in roi_cols:
        stat, p = ttest_ind(g0[roi], g1[roi])
        ttest_results[roi] = p

    ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index', columns=['p_value'])
    ttest_df = ttest_df.sort_values('p_value')

    reject, pvals_corrected, _, _ = multipletests(ttest_df['p_value'], method='fdr_bh')
    ttest_df['p_corrected'] = pvals_corrected
    ttest_df['significant'] = reject

    ttest_df.to_csv(os.path.join(OUTPUT_DIR, "PET_ttest_results.csv"))

    print("T-test done")

plt.figure(figsize=(12, 6))
sns.heatmap(roi_mean, cmap='coolwarm', center=0)
plt.title("ROI Mean per Subtype")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "PET_heatmap.png"))
plt.close()

print("Heatmap saved")

TOP_K = 10
top_rois = anova_df.head(TOP_K).index.tolist()

for roi in top_rois:
    plt.figure()
    sns.boxplot(x=SUBTYPE_COL, y=roi, data=df)
    plt.title(roi)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"PET_{roi}.png"))
    plt.close()

print("Boxplots saved")


def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        (np.std(x)**2 + np.std(y)**2) / 2
    )

if len(subtypes) == 2:
    effect_sizes = {}

    for roi in roi_cols:
        effect_sizes[roi] = cohen_d(g0[roi], g1[roi])

    effect_df = pd.DataFrame.from_dict(effect_sizes, orient='index', columns=['cohen_d'])
    effect_df = effect_df.reindex(effect_df['cohen_d'].abs().sort_values(ascending=False).index)

    effect_df.to_csv(os.path.join(OUTPUT_DIR, "PET_effect_size.csv"))

    print("Effect size saved")

print("\n Top significant ROIs:")
print(anova_df[anova_df['significant']].head(20))
