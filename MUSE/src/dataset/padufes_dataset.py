import os
import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import processed_data_path


class PADUFESDataset(Dataset):
    """
    Dataset for PAD-UFES-20 (image + tabular) with labels from metadata.csv ('diagnostic').

    Assumes the raw structure:
      {raw_data_path}/PAD-UFES-20/
        - images/<img_id>
        - metadata.csv
        - splits_padufes.json with keys: train/val/test each list of image filenames
    """

    def __init__(self,
                 split: str,
                 image_size: Tuple[int, int] = (224, 224),
                 tabular_cols: Optional[list] = None,
                 label_col: str = "diagnostic",
                 dev: bool = False,
                 load_no_label: bool = False):
        assert split in {"train", "val", "test"}
        if load_no_label:
            assert split == "train"
        self.split = split
        self.image_size = image_size
        self.label_col = label_col

        # Paths
        # Use the user-provided absolute data path
        self.data_dir = os.path.join("C:/Users/user/Desktop/연구/data", "PAD-UFES-20")
        self.images_dir = os.path.join(self.data_dir, "images")
        self.metadata_path = os.path.join(self.data_dir, "metadata.csv")
        self.splits_path = os.path.join(self.data_dir, "splits_padufes.json")

        # Load metadata and split
        self.meta = pd.read_csv(self.metadata_path)
        with open(self.splits_path, "r") as f:
            self.splits = json.load(f)
        self.image_ids = self.splits[split]
        if dev:
            self.image_ids = self.image_ids[:512]

        # Select columns for tabular; default: all numeric except label and id columns
        if tabular_cols is None:
            exclude = {self.label_col, "img_id"}
            numeric_cols = [c for c in self.meta.columns if c not in exclude]
            # Keep numeric-like; coerce non-numeric to NaN then fill
            self.tabular_cols = []
            for c in numeric_cols:
                try:
                    pd.to_numeric(self.meta[c])
                    self.tabular_cols.append(c)
                except Exception:
                    # skip non-numeric by default
                    pass
        else:
            self.tabular_cols = tabular_cols

        # Build label mapping
        labels = self.meta[self.label_col].dropna().unique().tolist()
        self.label2idx = {l: i for i, l in enumerate(sorted(labels))}

        # Basic transforms (use torchvision in collate or model)
        # We’ll handle normalization in dataset for simplicity
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, img_name: str) -> torch.Tensor:
        path = os.path.join(self.images_dir, img_name)
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize(self.image_size)
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
            x = (x - self.mean) / self.std
        return x

    def _row_for_image(self, img_name: str):
        row = self.meta[self.meta["img_id"] == img_name]
        if len(row) == 0:
            return None
        return row.iloc[0]

    def __getitem__(self, index: int):
        img_name = self.image_ids[index]
        row = self._row_for_image(img_name)

        # Image modality
        x_img_flag = True
        try:
            x_img = self._load_image(img_name)
        except Exception:
            x_img = torch.zeros(3, *self.image_size)
            x_img_flag = False

        # Tabular modality
        x_tab_flag = True
        if row is None or len(self.tabular_cols) == 0:
            x_tab = torch.zeros(len(self.tabular_cols)) if self.tabular_cols else torch.zeros(1)
            x_tab_flag = False
        else:
            vals = []
            for c in self.tabular_cols:
                v = row[c]
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                vals.append(v)
            arr = np.array(vals, dtype=np.float32)
            if np.isnan(arr).any():
                # simple impute
                arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)) if not np.isnan(np.nanmean(arr)) else 0.0)
            x_tab = torch.from_numpy(arr)

        # Label
        label_flag = True
        if (row is None) or (pd.isna(row[self.label_col])):
            y = 0
            label_flag = False
        else:
            y = int(self.label2idx[row[self.label_col]])

        return {
            "id": img_name,
            "x1": x_img,         # use x1 for image
            "x1_flag": x_img_flag,
            "x2": x_tab,         # use x2 for tabular
            "x2_flag": x_tab_flag,
            # x3 unused for this dataset
            "x3": torch.zeros(1),
            "x3_flag": False,
            "label": torch.tensor(y, dtype=torch.long),
            "label_flag": label_flag,
        }
