import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ROIMeanImputer:
    """
    Feature-wise mean imputer.
    Fit only on train data. Reuse for val/test.
    """
    def __init__(self):
        self.mean = None

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.mean = np.where(np.isnan(self.mean), 0.0, self.mean)
        return self

    def transform(self, X):
        X = X.copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.mean, inds[1])
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ROIZScore:
    """
    Feature-wise z-score scaler.
    Fit only on train data. Reuse for val/test.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def normalize_mri_by_total_roi_volume(X, eps=1e-8):
    """
    Normalize each subject's MRI ROI vector by the sum of that subject's
    matched MRI ROI volumes.

        X_norm[i, k] = X[i, k] / sum_j X[i, j]

    This is NOT true ICV normalization. It is relative regional-volume
    normalization using total matched MRI ROI volume as a proxy.
    """
    X = X.copy()
    total = X.sum(axis=1, keepdims=True)

    bad = (~np.isfinite(total)) | (total <= eps)
    if bad.any():
        total[bad] = 1.0

    return X / total


class ROIDataset(Dataset):
    def __init__(
        self,
        mri_df,
        pet_df,
        id_col="PTID",
        label_col="DX_bl",
        age_col="AGE",
        apoe_col="APOE4",
        age_mean=None,
        age_std=None,
        apoe_fill=None,
        mri_imputer=None,
        pet_imputer=None,
        mri_scaler=None,
        pet_scaler=None,
        label_encoder=None,
        fit=False,
    ):
        self.id_col = id_col
        self.label_col = label_col
        self.age_col = age_col
        self.apoe_col = apoe_col

        self.mri_cols = [c for c in mri_df.columns if "_VOLUME" in c.upper()]
        self.pet_cols = [c for c in pet_df.columns if "_SUVR" in c.upper()]

        required_mri_cols = [id_col, label_col, age_col, apoe_col] + self.mri_cols
        required_pet_cols = [id_col, label_col, age_col, apoe_col] + self.pet_cols

        mri_df = mri_df[required_mri_cols].copy()
        pet_df = pet_df[required_pet_cols].copy()

        mri_df = mri_df.sort_values(id_col).reset_index(drop=True)
        pet_df = pet_df.sort_values(id_col).reset_index(drop=True)

        assert (mri_df[id_col].values == pet_df[id_col].values).all(), \
            "PTID mismatch!"

        assert (mri_df[label_col].astype(str).values == pet_df[label_col].astype(str).values).all(), \
            "Label mismatch between MRI and PET!"

        self.ids = mri_df[id_col].values

        # -------------------------
        # Disease label
        # -------------------------
        y_raw = mri_df[label_col].astype(str).values

        if fit:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y_raw)
        else:
            assert label_encoder is not None, "Need label_encoder for val/test"
            self.label_encoder = label_encoder
            y = self.label_encoder.transform(y_raw)

        self.y = y.astype(np.int64)

        # -------------------------
        # Age target
        # -------------------------
        age_raw = pd.to_numeric(mri_df[age_col], errors="coerce").values.astype(np.float32)

        if fit:
            self.age_mean = np.nanmean(age_raw)
            self.age_std = np.nanstd(age_raw)

            if not np.isfinite(self.age_mean):
                self.age_mean = 0.0

            if (not np.isfinite(self.age_std)) or self.age_std == 0:
                self.age_std = 1.0
        else:
            assert age_mean is not None and age_std is not None, \
                "Need train age_mean and age_std for val/test"

            self.age_mean = age_mean
            self.age_std = age_std

        age_raw = np.where(np.isnan(age_raw), self.age_mean, age_raw)
        self.age = ((age_raw - self.age_mean) / self.age_std).astype(np.float32)

        # -------------------------
        # APOE4 target
        # -------------------------
        apoe_raw = pd.to_numeric(mri_df[apoe_col], errors="coerce").values

        if fit:
            valid_apoe = apoe_raw[np.isfinite(apoe_raw)]

            if len(valid_apoe) == 0:
                self.apoe_fill = 0
            else:
                values, counts = np.unique(valid_apoe.astype(np.int64), return_counts=True)
                self.apoe_fill = int(values[np.argmax(counts)])
        else:
            if apoe_fill is None:
                apoe_fill = 0
            self.apoe_fill = int(apoe_fill)

        apoe_raw = np.where(np.isnan(apoe_raw), self.apoe_fill, apoe_raw)
        self.apoe = apoe_raw.astype(np.int64)

        # Safety check for APOE4 classes.
        bad_apoe = ~np.isin(self.apoe, [0, 1, 2])
        if bad_apoe.any():
            raise ValueError(
                f"APOE4 contains values outside {{0,1,2}}: "
                f"{np.unique(self.apoe[bad_apoe])}"
            )

        # -------------------------
        # ROI matrices
        # -------------------------
        X_mri = (
            mri_df[self.mri_cols]
            .apply(pd.to_numeric, errors="coerce")
            .values
            .astype(np.float32)
        )

        X_pet = (
            pet_df[self.pet_cols]
            .apply(pd.to_numeric, errors="coerce")
            .values
            .astype(np.float32)
        )

        self.mri_missing_rate_before_impute = np.isnan(X_mri).mean()
        self.pet_missing_rate_before_impute = np.isnan(X_pet).mean()

        if fit:
            # MRI: raw -> impute -> total-volume normalize -> z-score
            # PET: raw -> impute -> z-score
            self.mri_imputer = ROIMeanImputer()
            self.pet_imputer = ROIMeanImputer()

            X_mri = self.mri_imputer.fit_transform(X_mri)
            X_pet = self.pet_imputer.fit_transform(X_pet)

            X_mri = normalize_mri_by_total_roi_volume(X_mri)

            self.mri_total_normalized_sum_min = X_mri.sum(axis=1).min()
            self.mri_total_normalized_sum_max = X_mri.sum(axis=1).max()

            self.mri_scaler = ROIZScore()
            self.pet_scaler = ROIZScore()

            X_mri = self.mri_scaler.fit_transform(X_mri)
            X_pet = self.pet_scaler.fit_transform(X_pet)

        else:
            assert mri_imputer is not None and pet_imputer is not None, \
                "Must provide train imputers for val/test"
            assert mri_scaler is not None and pet_scaler is not None, \
                "Must provide train scalers for val/test"

            self.mri_imputer = mri_imputer
            self.pet_imputer = pet_imputer
            self.mri_scaler = mri_scaler
            self.pet_scaler = pet_scaler

            X_mri = self.mri_imputer.transform(X_mri)
            X_pet = self.pet_imputer.transform(X_pet)

            X_mri = normalize_mri_by_total_roi_volume(X_mri)

            self.mri_total_normalized_sum_min = X_mri.sum(axis=1).min()
            self.mri_total_normalized_sum_max = X_mri.sum(axis=1).max()

            X_mri = self.mri_scaler.transform(X_mri)
            X_pet = self.pet_scaler.transform(X_pet)

        self.X_mri = X_mri.astype(np.float32)
        self.X_pet = X_pet.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_mri[idx]),
            torch.tensor(self.X_pet[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.apoe[idx]),
            torch.tensor(self.age[idx]),
        )