import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ROIZScore:
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

class ROIDataset(Dataset):
    def __init__(
        self,
        mri_df,
        pet_df,
        id_col="PTID",
        mri_scaler=None,
        pet_scaler=None,
        fit=False
    ):
        self.id_col = id_col

        self.mri_cols = [c for c in mri_df.columns if "_VOLUME" in c.upper()]
        self.pet_cols = [c for c in pet_df.columns if "_SUVR" in c.upper()]

        mri_df = mri_df[[id_col] + self.mri_cols].copy()
        pet_df = pet_df[[id_col] + self.pet_cols].copy()

        mri_df = mri_df.sort_values(id_col).reset_index(drop=True)
        pet_df = pet_df.sort_values(id_col).reset_index(drop=True)

        assert (mri_df[id_col].values == pet_df[id_col].values).all(), \
            "PTID mismatch!"

        self.ids = mri_df[id_col].values

        X_mri = mri_df[self.mri_cols].apply(pd.to_numeric, errors="coerce").values
        X_pet = pet_df[self.pet_cols].apply(pd.to_numeric, errors="coerce").values

        X_mri = np.nan_to_num(X_mri).astype(np.float32)
        X_pet = np.nan_to_num(X_pet).astype(np.float32)

        if fit:
            self.mri_scaler = ROIZScore()
            self.pet_scaler = ROIZScore()

            X_mri = self.mri_scaler.fit_transform(X_mri)
            X_pet = self.pet_scaler.fit_transform(X_pet)

        else:
            assert mri_scaler is not None and pet_scaler is not None, \
                "Must provide train scalers for val/test"

            self.mri_scaler = mri_scaler
            self.pet_scaler = pet_scaler

            X_mri = self.mri_scaler.transform(X_mri)
            X_pet = self.pet_scaler.transform(X_pet)

        self.X_mri = X_mri.astype(np.float32)
        self.X_pet = X_pet.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_mri[idx]),
            torch.tensor(self.X_pet[idx])
        )