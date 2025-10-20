import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class YELPDataset(Dataset):
    """
    Yelp Open Dataset for photo-level 3-modality classification.

    Instance = photo_id from photos.json

    - x1 (image): yelp_photos/photos/{photo_id}.jpg
    - x2 (tabular): features aggregated from business.json (joined by business_id)
    - x3 (text): photos.json caption; fallback to empty string
    - y (label): photos.json label in {food, inside, outside, drink, menu}
    """

    LABELS = ["drink", "food", "inside", "menu", "outside"]  # sorted for stability

    def __init__(
        self,
        split: str = "train",
        root_dir: str = "C:/Users/user/Desktop/연구/data/Yelp",
        image_size: Tuple[int, int] = (224, 224),
        dev: bool = False,
        max_samples: Optional[int] = None,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.root_dir = root_dir
        self.image_size = image_size

        # Paths
        self.photos_dir = os.path.join(root_dir, "yelp_photos", "photos")
        self.photos_json = os.path.join(root_dir, "yelp_photos", "photos.json")
        self.business_json = os.path.join(root_dir, "yelp_json", "yelp_academic_dataset_business.json")

        # Load photos metadata: support JSON array or JSONL. Fallback to Yelp's default filename if needed.
        photos = []
        def _read_json_or_jsonl(path: str) -> List[Dict]:
            recs: List[Dict] = []
            if not os.path.exists(path):
                return recs
            with open(path, "r", encoding="utf-8") as f:
                try:
                    obj = json.load(f)
                    if isinstance(obj, list):
                        return obj
                    elif isinstance(obj, dict):
                        return [obj]
                except json.JSONDecodeError:
                    # JSONL fallback
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            recs.append(json.loads(line))
                        except Exception:
                            continue
            return recs

        photos = _read_json_or_jsonl(self.photos_json)
        if not photos:
            # Default Yelp academic filename location (JSONL)
            alt_photos = os.path.join(root_dir, "yelp_json", "yelp_academic_dataset_photo.json")
            photos = _read_json_or_jsonl(alt_photos)

        # Load business JSON (JSONL) -> DataFrame
        biz_rows: List[Dict] = []
        with open(self.business_json, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    biz_rows.append(json.loads(line))
                except Exception:
                    continue
        self.biz_df = pd.DataFrame(biz_rows)
        self.biz_df.set_index("business_id", inplace=True)

        # Build items with availability flags
        items: List[Dict] = []
        for p in photos:
            photo_id = p.get("photo_id")
            business_id = p.get("business_id")
            label = p.get("label")
            caption = p.get("caption") or ""
            if (photo_id is None) or (business_id is None) or (label is None):
                continue
            img_path = os.path.join(self.photos_dir, f"{photo_id}.jpg")
            if not os.path.exists(img_path):
                continue
            if business_id not in self.biz_df.index:
                continue
            items.append({
                "photo_id": photo_id,
                "business_id": business_id,
                "label_str": label,
                "caption": caption,
                "img_path": img_path,
            })

        # Basic split: deterministic shard by hash(photo_id)
        def split_bucket(pid: str) -> str:
            h = abs(hash(pid)) % 100
            if h < 80:
                return "train"
            elif h < 90:
                return "val"
            return "test"

        items = [it for it in items if split_bucket(it["photo_id"]) == split]
        if dev:
            items = items[:2000]
        if max_samples:
            items = items[:max_samples]
        self.items = items

        # Label mapping
        self.label2idx = {lbl: i for i, lbl in enumerate(sorted(self.LABELS))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # Image normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Precompute minimal tabular feature names (numeric-friendly)
        self.tabular_cols = [
            "stars", "review_count",
            # Location (can be helpful)
            "latitude", "longitude",
            # Simple binary/open flag
            "is_open",
        ]
        # Missing columns are filled with 0

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize(self.image_size)
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
            x = (x - self.mean) / self.std
        return x

    def _tabular_from_business(self, business_id: str) -> torch.Tensor:
        if business_id not in self.biz_df.index:
            return torch.zeros(len(self.tabular_cols), dtype=torch.float32)
        row = self.biz_df.loc[business_id]
        vals: List[float] = []
        for c in self.tabular_cols:
            try:
                v = float(row.get(c, 0.0))
            except Exception:
                v = 0.0
            if np.isnan(v):
                v = 0.0
            vals.append(v)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        photo_id = it["photo_id"]
        bid = it["business_id"]
        label_str = it["label_str"]
        caption = it["caption"]
        img_path = it["img_path"]

        # Image
        x1_flag = True
        try:
            x1 = self._load_image(img_path)
        except Exception:
            x1 = torch.zeros(3, *self.image_size)
            x1_flag = False

        # Tabular
        x2 = self._tabular_from_business(bid)
        x2_flag = x2.numel() > 0

        # Text (caption)
        text = caption if isinstance(caption, str) else ""
        x3_flag = len(text) > 0

        # Label
        label = self.label2idx.get(label_str, 0)
        label_flag = True

        return {
            "id": photo_id,
            "x1": x1,
            "x1_flag": x1_flag,
            "x2": x2,
            "x2_flag": x2_flag,
            "x3": text,
            "x3_flag": x3_flag,
            "label": torch.tensor(label, dtype=torch.long),
            "label_flag": label_flag,
        }
