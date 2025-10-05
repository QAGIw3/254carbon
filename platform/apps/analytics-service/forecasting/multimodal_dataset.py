"""Dataset utilities for multimodal, multi-commodity transformer training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class MultiModalSample:
    """Structured sample returned by the dataset."""

    commodities: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
    targets: Dict[str, torch.Tensor]
    time_index: pd.DatetimeIndex
    start_idx: int


class MultiModalDataset(Dataset):
    """Slide window dataset that exposes aligned multimodal inputs per commodity."""

    def __init__(
        self,
        feature_bundle: Dict[str, Dict[str, object]],
        *,
        seq_len: int,
        forecast_horizons: Sequence[int],
        target_modality: str = "price",
        price_target_column: Optional[str] = None,
    ) -> None:
        if not feature_bundle:
            raise ValueError("feature_bundle must contain at least one commodity")

        self.feature_bundle = feature_bundle
        self.seq_len = int(seq_len)
        self.forecast_horizons = sorted(int(h) for h in forecast_horizons)
        self.max_horizon = max(self.forecast_horizons)
        self.target_modality = target_modality
        self.price_target_column = price_target_column

        self.commodities = sorted(feature_bundle.keys())
        self._modalities = self._collect_modalities(feature_bundle)
        self._series_store = self._prepare_arrays(feature_bundle)
        self._mask_store = self._prepare_arrays(feature_bundle, masks=True)
        self._targets = self._prepare_targets(feature_bundle)
        self._time_index = self._extract_time_index(feature_bundle)

        self._normalization_stats = self._compute_normalization_stats()
        self._num_samples = max(0, len(self._time_index) - self.seq_len - self.max_horizon + 1)
        if self._num_samples <= 0:
            raise ValueError(
                "Insufficient timesteps to create samples; "
                f"need >= seq_len + max_horizon ({self.seq_len + self.max_horizon})"
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> MultiModalSample:
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(idx)

        start = idx
        end = idx + self.seq_len

        sample_modalities: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        targets: Dict[str, torch.Tensor] = {}

        for commodity in self.commodities:
            commodity_modalities: Dict[str, Dict[str, torch.Tensor]] = {
                "modalities": {},
                "masks": {},
            }

            for modality in self._modalities:
                data = self._series_store[commodity][modality]
                mask = self._mask_store[commodity][modality]
                window = data[start:end]
                mask_window = mask[start:end]
                mean, std = self._normalization_stats[commodity][modality]
                window = (window - mean) / std
                commodity_modalities["modalities"][modality] = torch.from_numpy(window)
                commodity_modalities["masks"][modality] = torch.from_numpy(mask_window)

            targets[commodity] = self._slice_targets(commodity, end)
            sample_modalities[commodity] = commodity_modalities

        time_index = self._time_index[start:end]

        return MultiModalSample(
            commodities=sample_modalities,
            targets=targets,
            time_index=time_index,
            start_idx=start,
        )

    # ------------------------------------------------------------------
    # Collation helper
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: Sequence[MultiModalSample]) -> Dict[str, object]:
        if not batch:
            raise ValueError("Empty batch provided to collate_fn")

        commodities = list(batch[0].commodities.keys())
        modalities = list(batch[0].commodities[commodities[0]]["modalities"].keys())

        collated: Dict[str, object] = {
            "commodities": {},
            "time_index": [sample.time_index for sample in batch],
            "start_indices": torch.tensor([sample.start_idx for sample in batch], dtype=torch.long),
        }

        for commodity in commodities:
            modality_tensors: Dict[str, torch.Tensor] = {}
            mask_tensors: Dict[str, torch.Tensor] = {}
            target_tensors: List[torch.Tensor] = []

            for modality in modalities:
                stacked = torch.stack(
                    [sample.commodities[commodity]["modalities"][modality] for sample in batch],
                    dim=0,
                )
                mask_stacked = torch.stack(
                    [sample.commodities[commodity]["masks"][modality] for sample in batch],
                    dim=0,
                )
                modality_tensors[modality] = stacked
                mask_tensors[modality] = mask_stacked

            for sample in batch:
                target_tensors.append(sample.targets[commodity])

            collated["commodities"][commodity] = {
                "modalities": modality_tensors,
                "masks": mask_tensors,
                "targets": torch.stack(target_tensors, dim=0),
            }

        return collated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_modalities(self, feature_bundle: Dict[str, Dict[str, object]]) -> List[str]:
        modality_set = set()
        for payload in feature_bundle.values():
            modality_set.update(payload.get("modalities", {}).keys())
        if self.target_modality not in modality_set:
            raise ValueError(
                f"Target modality '{self.target_modality}' missing from feature bundle."
            )
        return sorted(modality_set)

    def _prepare_arrays(
        self,
        feature_bundle: Dict[str, Dict[str, object]],
        masks: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        store: Dict[str, Dict[str, np.ndarray]] = {}
        for commodity, payload in feature_bundle.items():
            store[commodity] = {}
            source = "masks" if masks else "modalities"
            entries: Dict[str, pd.DataFrame] = payload.get(source, {})
            for modality in self._modalities:
                frame = entries.get(modality)
                if frame is None or frame.empty:
                    length = len(payload.get("time_index", []))
                    store[commodity][modality] = np.zeros((length, 0), dtype=np.float32)
                else:
                    values = frame.values.astype(np.float32)
                    if values.ndim == 1:
                        values = values.reshape(-1, 1)
                    store[commodity][modality] = values
        return store

    def _prepare_targets(self, feature_bundle: Dict[str, Dict[str, object]]) -> Dict[str, np.ndarray]:
        targets: Dict[str, np.ndarray] = {}
        for commodity, payload in feature_bundle.items():
            modality_frames: Dict[str, pd.DataFrame] = payload.get("modalities", {})
            target_frame = modality_frames.get(self.target_modality)
            if target_frame is None or target_frame.empty:
                length = len(payload.get("time_index", []))
                targets[commodity] = np.zeros(length, dtype=np.float32)
                continue

            if self.price_target_column and self.price_target_column in target_frame.columns:
                target_series = target_frame[self.price_target_column]
            else:
                target_series = target_frame.iloc[:, 0]
            targets[commodity] = target_series.values.astype(np.float32)
        return targets

    def _extract_time_index(self, feature_bundle: Dict[str, Dict[str, object]]) -> pd.DatetimeIndex:
        first = next(iter(feature_bundle.values()))
        index = first.get("time_index")
        if index is None:
            raise ValueError("Feature bundle entries must include a time_index")
        for payload in feature_bundle.values():
            other = payload.get("time_index")
            if not other.equals(index):
                raise ValueError("All commodities must share the same time index")
        return index

    def _compute_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        stats: Dict[str, Dict[str, np.ndarray]] = {}
        for commodity in self.commodities:
            stats[commodity] = {}
            for modality in self._modalities:
                data = self._series_store[commodity][modality]
                if data.size == 0:
                    stats[commodity][modality] = (
                        np.zeros((1, 0), dtype=np.float32),
                        np.ones((1, 0), dtype=np.float32),
                    )
                    continue
                mean = np.nanmean(data, axis=0, keepdims=True)
                std = np.nanstd(data, axis=0, keepdims=True)
                std = np.where(std < 1e-6, 1.0, std)
                stats[commodity][modality] = (
                    mean.astype(np.float32),
                    std.astype(np.float32),
                )
        return stats

    def _slice_targets(self, commodity: str, anchor: int) -> torch.Tensor:
        series = self._targets[commodity]
        horizon_values: List[float] = []
        for horizon in self.forecast_horizons:
            target_idx = anchor + horizon - 1
            if target_idx >= len(series):
                raise IndexError(
                    f"Target index {target_idx} out of range for commodity {commodity}"
                )
            horizon_values.append(series[target_idx])
        return torch.tensor(horizon_values, dtype=torch.float32)


def multimodal_collate(batch: Sequence[MultiModalSample]) -> Dict[str, object]:
    """Wrapper to access the default collate function from external modules."""
    return MultiModalDataset.collate_fn(batch)
