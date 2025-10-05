"""
Minimal fallback deep learning module.

Provides a stub implementation of MultiCommodityTransformer sufficient for
service startup and basic inference scaffolding. This is NOT a production
model; it returns simple constant statistics to satisfy code paths.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn


class MultiCommodityTransformer(nn.Module):
    def __init__(
        self,
        *,
        commodity_modalities: Dict[str, Dict[str, int]],
        commodity_groups: Dict[str, str],
        d_model: int,
        num_heads: int,
        num_layers: int,
        cross_commodity_layers: int,
        d_ff: int,
        dropout: float,
        forecast_horizons: List[int],
        head_mask_config: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        super().__init__()
        self.commodity_modalities = commodity_modalities
        self.commodity_groups = commodity_groups
        self.horizons = list(forecast_horizons)

        # Lightweight per-commodity fusion gate parameters (one per modality)
        self.fusion_params: nn.ModuleDict = nn.ModuleDict()
        for commodity, modalities in commodity_modalities.items():
            self.fusion_params[commodity] = nn.ParameterDict(
                {modality: nn.Parameter(torch.tensor(0.5)) for modality in modalities.keys()}
            )

    def forward(
        self,
        inputs: Dict[str, Dict[str, torch.Tensor]],
        *,
        commodity_masks: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        return_attentions: bool = False,
    ) -> Dict[str, Dict[str, Dict]]:
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")
        forecasts: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        fusion_gates: Dict[str, Dict[str, torch.Tensor]] = {}

        for commodity, modality_inputs in inputs.items():
            # Derive batch size from the first modality tensor
            first_tensor = next(iter(modality_inputs.values()))
            batch_size = first_tensor.shape[0]

            # Build trivial forecasts: zeros for mean and small positive std
            horizon_map: Dict[int, Dict[str, torch.Tensor]] = {}
            for horizon in self.horizons:
                mean = torch.zeros(batch_size, device=device)
                std = torch.full((batch_size,), 0.1, device=device)
                horizon_map[int(horizon)] = {"mean": mean, "std": std}
            forecasts[commodity] = horizon_map

            # Expose simplistic fusion gates per modality
            gates = {}
            if commodity in self.fusion_params:
                for modality, param in self.fusion_params[commodity].items():
                    gates[modality] = param.detach().to(device)
            else:
                for modality in modality_inputs.keys():
                    gates[modality] = torch.tensor(0.5, device=device)
            fusion_gates[commodity] = gates

        output = {"forecasts": forecasts, "fusion_gates": fusion_gates}
        if return_attentions:
            # Provide an empty attention structure to satisfy callers
            output["cross_attentions"] = []
        return output


