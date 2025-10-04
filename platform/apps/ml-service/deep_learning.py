"""
Deep Learning Models for Price Forecasting

Implements transformer-based models with attention mechanisms
for multi-horizon forecasting.
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Injects information about the relative position of tokens.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[torch.Tensor] = None
    ):
        """Apply multi-head attention."""
        batch_size = query.size(0)
        
        # Linear projections in batch
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection and attention weights
        return self.W_o(x), attention


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """Apply transformer block."""
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class ModalityEncoder(nn.Module):
    """Generic encoder for a single data modality."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.d_model = d_model
        self.enabled = self.input_dim > 0

        if not self.enabled:
            self.register_buffer("_fallback", torch.zeros(1))
            return

        self.input_projection = nn.Linear(self.input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled or x.size(-1) == 0:
            batch, seq_len = x.size(0), x.size(1)
            return torch.zeros(batch, seq_len, self.d_model, device=x.device, dtype=x.dtype)

        h = self.input_projection(x)
        h = self.positional(h)
        h = self.dropout(h)
        return self.encoder(h, src_key_padding_mask=key_padding_mask)


class GatedModalityFusion(nn.Module):
    """Fuse modality embeddings using learned gates."""

    def __init__(
        self,
        modalities: List[str],
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.d_model = d_model
        self.gates = nn.Parameter(torch.ones(len(self.modalities)))
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        available_indices: List[int] = []
        stacked_embeddings: List[torch.Tensor] = []

        for idx, modality in enumerate(self.modalities):
            tensor = embeddings.get(modality)
            if tensor is None:
                continue
            available_indices.append(idx)
            stacked_embeddings.append(tensor)

        if not stacked_embeddings:
            raise ValueError("No modality embeddings provided for fusion")

        gate_logits = self.gates[available_indices]
        weights = torch.softmax(gate_logits, dim=0)

        weighted_components = [
            weight.view(1, 1, 1) * tensor
            for weight, tensor in zip(weights, stacked_embeddings)
        ]
        fused = torch.stack(weighted_components, dim=0).sum(dim=0)
        fused = self.projection(self.dropout(fused))

        weight_dict = {
            self.modalities[idx]: weight
            for idx, weight in zip(available_indices, weights)
        }
        return fused, weight_dict


class CommodityHeadGatedAttention(nn.Module):
    """Multi-head attention with per-commodity head gating."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_commodities: int,
        dropout: float = 0.1,
        enable_gating: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_commodities = num_commodities
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if enable_gating:
            self.head_gate = nn.Parameter(torch.ones(num_commodities, num_heads))
        else:
            self.register_parameter("head_gate", None)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        commodity_idx: int,
        key_padding_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len = query.size(0), query.size(1)

        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,K)
            scores = scores.masked_fill(expanded_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        if self.head_gate is not None:
            gate = torch.softmax(self.head_gate[commodity_idx], dim=-1)
            attn_weights = attn_weights * gate.view(1, self.num_heads, 1, 1)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask.view(1, self.num_heads, 1, 1)

        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        output = self.out_proj(context)

        return output, attn_weights


class CrossCommodityBlock(nn.Module):
    """Cross-commodity attention block with optional head gating."""

    def __init__(
        self,
        num_commodities: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        enable_head_gating: bool = True,
    ) -> None:
        super().__init__()
        self.num_commodities = num_commodities
        self.attention = CommodityHeadGatedAttention(
            d_model,
            num_heads,
            num_commodities,
            dropout=dropout,
            enable_gating=enable_head_gating,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        states: torch.Tensor,
        *,
        adjacency_mask: Optional[torch.Tensor] = None,
        head_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs: List[torch.Tensor] = []
        attn_maps: List[torch.Tensor] = []
        batch = states.size(0)

        for idx in range(self.num_commodities):
            query = states[:, idx : idx + 1, :]
            key_padding_mask = None
            if adjacency_mask is not None:
                row = adjacency_mask[idx].to(dtype=torch.bool)
                allowed = row.unsqueeze(0).expand(batch, -1)
                key_padding_mask = ~allowed

            head_mask = None
            if head_masks is not None and head_masks[idx] is not None:
                head_mask = head_masks[idx].to(states.device)

            attn_output, attn_weights = self.attention(
                query,
                states,
                states,
                commodity_idx=idx,
                key_padding_mask=key_padding_mask,
                head_mask=head_mask,
            )

            residual = self.norm1(query + attn_output)
            ff_out = self.ffn(residual)
            final = self.norm2(residual + ff_out)

            outputs.append(final)
            attn_maps.append(attn_weights)

        updated_states = torch.cat(outputs, dim=1)
        attention_tensor = torch.stack(attn_maps, dim=1)
        return updated_states, attention_tensor


class PriceForecaster(nn.Module):
    """
    Transformer-based price forecasting model.
    
    Multi-horizon forecasting with uncertainty quantification.
    """
    
    def __init__(
        self,
        input_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 168,  # 1 week of hourly data
        forecast_horizons: List[int] = [1, 6, 24, 168],  # 1h, 6h, 1d, 1w
    ):
        super().__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        self.forecast_horizons = forecast_horizons
        
        # Input embedding
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Multi-horizon output heads
        self.output_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, 2),  # Mean and std for uncertainty
            )
            for h in forecast_horizons
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            mask: Optional attention mask
        
        Returns:
            Dictionary of forecasts by horizon with mean and std
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Take the last timestep's representation
        x = x[:, -1, :]
        
        # Generate multi-horizon forecasts
        forecasts = {}
        for horizon in self.forecast_horizons:
            output = self.output_heads[f"horizon_{horizon}"](x)
            forecasts[horizon] = {
                "mean": output[:, 0],
                "std": torch.exp(output[:, 1]),  # Ensure positive std
            }
        
        return forecasts


class MultiCommodityTransformer(nn.Module):
    """Enhanced multimodal transformer for cross-commodity forecasting."""

    def __init__(
        self,
        commodity_modalities: Dict[str, Dict[str, int]],
        *,
        commodity_groups: Optional[Dict[str, str]] = None,
        modality_settings: Optional[Dict[str, Dict[str, int]]] = None,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        cross_commodity_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 365,
        forecast_horizons: Optional[List[int]] = None,
        enable_head_gating: bool = True,
        market_graph: Optional[Dict[str, List[str]]] = None,
        head_mask_config: Optional[Dict[str, List[int]]] = None,
        primary_modality: str = "price",
    ) -> None:
        super().__init__()
        if not commodity_modalities:
            raise ValueError("commodity_modalities must not be empty")

        self.commodity_order = list(commodity_modalities.keys())
        self.num_commodities = len(self.commodity_order)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_layer = nn.Dropout(dropout)
        self.forecast_horizons = sorted(forecast_horizons or [1, 7, 30, 90])
        self.primary_modality = primary_modality
        self.enable_head_gating = enable_head_gating

        base_modality_settings = {
            "price": {
                "num_layers": max(2, num_layers // 2),
                "num_heads": min(num_heads, 4),
                "d_ff": max(128, d_ff // 2),
            },
            "fundamentals": {
                "num_layers": max(1, num_layers // 3),
                "num_heads": min(num_heads, 4),
                "d_ff": max(128, d_ff // 2),
            },
            "weather": {
                "num_layers": max(1, num_layers // 3),
                "num_heads": min(num_heads, 4),
                "d_ff": max(128, d_ff // 2),
            },
        }
        if modality_settings:
            for modality, override in modality_settings.items():
                base_modality_settings.setdefault(modality, {})
                base_modality_settings[modality].update(override)

        self.modality_encoders = nn.ModuleDict()
        self.modality_fusion = nn.ModuleDict()
        self.commodity_transformers = nn.ModuleDict()
        self.commodity_norms = nn.ModuleDict()

        for commodity in self.commodity_order:
            modality_dims = commodity_modalities[commodity]
            commodity_encoder_dict = nn.ModuleDict()
            for modality, feature_dim in modality_dims.items():
                settings = base_modality_settings.get(modality, base_modality_settings["price"])
                commodity_encoder_dict[modality] = ModalityEncoder(
                    input_dim=feature_dim,
                    d_model=d_model,
                    num_layers=settings.get("num_layers", 2),
                    num_heads=settings.get("num_heads", min(num_heads, 4)),
                    d_ff=settings.get("d_ff", max(128, d_ff // 2)),
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
            self.modality_encoders[commodity] = commodity_encoder_dict
            self.modality_fusion[commodity] = GatedModalityFusion(
                list(modality_dims.keys()), d_model, dropout
            )
            self.commodity_transformers[commodity] = nn.ModuleList(
                [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
            )
            self.commodity_norms[commodity] = nn.LayerNorm(d_model)

        self.cross_blocks = nn.ModuleList(
            [
                CrossCommodityBlock(
                    num_commodities=self.num_commodities,
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    enable_head_gating=enable_head_gating,
                )
                for _ in range(cross_commodity_layers)
            ]
        )

        self.output_heads = nn.ModuleDict(
            {
                commodity: nn.ModuleDict(
                    {
                        f"horizon_{h}": nn.Sequential(
                            nn.Linear(d_model, d_ff),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_ff, 2),
                        )
                        for h in self.forecast_horizons
                    }
                )
                for commodity in self.commodity_order
            }
        )

        self.register_buffer(
            "_market_graph_mask",
            self._build_market_graph_mask(market_graph, commodity_groups),
            persistent=False,
        )
        self.head_masks = self._build_head_masks(
            head_mask_config,
            commodity_groups,
            num_heads,
        )

    def forward(
        self,
        commodity_inputs: Dict[str, Dict[str, torch.Tensor]],
        *,
        commodity_masks: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        market_graph: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Dict[str, Any]:
        commodity_masks = commodity_masks or {}
        device = next(self.parameters()).device

        fusion_summaries: Dict[str, Dict[str, torch.Tensor]] = {}
        pooled_states: List[torch.Tensor] = []

        for commodity in self.commodity_order:
            if commodity not in commodity_inputs:
                raise ValueError(f"Missing inputs for commodity '{commodity}'")

            inputs = commodity_inputs[commodity]
            masks = commodity_masks.get(commodity, {})

            if not inputs:
                raise ValueError(f"No modality inputs provided for commodity '{commodity}'")

            sample_tensor = next(iter(inputs.values()))
            batch_size, seq_len = sample_tensor.size(0), sample_tensor.size(1)
            dtype = sample_tensor.dtype

            encoded_modalities: Dict[str, torch.Tensor] = {}

            for modality, encoder in self.modality_encoders[commodity].items():
                tensor = inputs.get(modality)
                if tensor is None:
                    tensor = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
                tensor = tensor.to(device)

                mask_tensor = masks.get(modality)
                key_padding_mask = None
                if mask_tensor is not None and mask_tensor.numel() > 0:
                    key_padding_mask = (mask_tensor.to(device).sum(dim=-1) == 0)

                encoded_modalities[modality] = encoder(
                    tensor,
                    key_padding_mask=key_padding_mask,
                )

            fused_sequence, fusion_weights = self.modality_fusion[commodity](
                encoded_modalities,
                masks=masks,
            )

            sequence = fused_sequence
            for block in self.commodity_transformers[commodity]:
                sequence = block(sequence)
            sequence = self.commodity_norms[commodity](sequence)

            fusion_summaries[commodity] = fusion_weights
            pooled_states.append(sequence[:, -1, :])

        commodity_states = torch.stack(pooled_states, dim=1)
        adjacency = self._resolve_market_graph(market_graph).to(commodity_states.device)

        head_masks = None
        if self.enable_head_gating and self.head_masks:
            head_masks = [
                mask.to(commodity_states.device) if mask is not None else None
                for mask in self.head_masks
            ]

        cross_attentions: List[torch.Tensor] = []
        for block in self.cross_blocks:
            commodity_states, attn = block(
                commodity_states,
                adjacency_mask=adjacency,
                head_masks=head_masks,
            )
            cross_attentions.append(attn)

        commodity_states = self.dropout_layer(commodity_states)

        forecasts: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        for idx, commodity in enumerate(self.commodity_order):
            context = commodity_states[:, idx, :]
            commodity_outputs: Dict[int, Dict[str, torch.Tensor]] = {}
            for horizon in self.forecast_horizons:
                head = self.output_heads[commodity][f"horizon_{horizon}"]
                out = head(context)
                mean = out[:, 0]
                log_var = out[:, 1]
                std = torch.exp(0.5 * log_var)
                commodity_outputs[horizon] = {
                    "mean": mean,
                    "std": std,
                    "log_var": log_var,
                }
            forecasts[commodity] = commodity_outputs

        result: Dict[str, Any] = {
            "forecasts": forecasts,
            "fusion_gates": fusion_summaries,
        }
        if return_attentions:
            result["cross_attentions"] = cross_attentions
        return result

    def enable_dropout_for_uncertainty(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        commodity_inputs: Dict[str, Dict[str, torch.Tensor]],
        *,
        commodity_masks: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        market_graph: Optional[torch.Tensor] = None,
        n_samples: int = 50,
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        self.enable_dropout_for_uncertainty()
        self.eval()

        sample_outputs: List[Dict[str, Dict[int, Dict[str, torch.Tensor]]]] = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(
                    commodity_inputs,
                    commodity_masks=commodity_masks,
                    market_graph=market_graph,
                    return_attentions=False,
                )
                sample_outputs.append(outputs["forecasts"])

        aggregated: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        for commodity in self.commodity_order:
            commodity_summary: Dict[int, Dict[str, torch.Tensor]] = {}
            for horizon in self.forecast_horizons:
                horizon_means = torch.stack(
                    [sample[commodity][horizon]["mean"] for sample in sample_outputs], dim=0
                )
                horizon_stds = torch.stack(
                    [sample[commodity][horizon]["std"] for sample in sample_outputs], dim=0
                )

                mean_prediction = horizon_means.mean(dim=0)
                aleatoric = horizon_stds.mean(dim=0)
                epistemic = horizon_means.std(dim=0)
                total_uncertainty = torch.sqrt(aleatoric ** 2 + epistemic ** 2)

                commodity_summary[horizon] = {
                    "mean": mean_prediction,
                    "std": aleatoric,
                    "epistemic_uncertainty": epistemic,
                    "total_uncertainty": total_uncertainty,
                    "prediction_interval_95": (
                        mean_prediction - 1.96 * total_uncertainty,
                        mean_prediction + 1.96 * total_uncertainty,
                    ),
                }
            aggregated[commodity] = commodity_summary

        return aggregated

    def _build_market_graph_mask(
        self,
        market_graph: Optional[Dict[str, List[str]]],
        commodity_groups: Optional[Dict[str, str]],
    ) -> torch.Tensor:
        adjacency = torch.ones(self.num_commodities, self.num_commodities, dtype=torch.bool)
        if market_graph:
            adjacency = torch.zeros_like(adjacency)
            index_map = {name: idx for idx, name in enumerate(self.commodity_order)}
            for commodity, neighbors in market_graph.items():
                idx = index_map.get(commodity)
                if idx is None:
                    continue
                adjacency[idx, idx] = True
                for neighbor in neighbors:
                    neighbor_idx = index_map.get(neighbor)
                    if neighbor_idx is not None:
                        adjacency[idx, neighbor_idx] = True
        adjacency = adjacency | torch.eye(self.num_commodities, dtype=torch.bool)
        return adjacency

    def _build_head_masks(
        self,
        head_mask_config: Optional[Dict[str, List[int]]],
        commodity_groups: Optional[Dict[str, str]],
        num_heads: int,
    ) -> List[Optional[torch.Tensor]]:
        if not head_mask_config:
            return [None] * self.num_commodities

        head_masks: List[Optional[torch.Tensor]] = []
        for commodity in self.commodity_order:
            allowed = head_mask_config.get(commodity)
            if allowed is None and commodity_groups:
                group = commodity_groups.get(commodity)
                if group:
                    allowed = head_mask_config.get(group)
            if not allowed:
                head_masks.append(None)
                continue
            mask = torch.zeros(num_heads, dtype=torch.float32)
            for head_idx in allowed:
                if 0 <= head_idx < num_heads:
                    mask[head_idx] = 1.0
            head_masks.append(mask)
        return head_masks

    def _resolve_market_graph(self, market_graph: Optional[torch.Tensor]) -> torch.Tensor:
        if market_graph is None:
            return self._market_graph_mask
        if isinstance(market_graph, torch.Tensor):
            return market_graph.to(dtype=torch.bool)
        raise TypeError("market_graph must be a tensor or None")


class PriceDataset(Dataset):
    """Dataset for price forecasting."""
    
    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        seq_len: int = 168,
        forecast_horizons: List[int] = [1, 6, 24, 168],
    ):
        self.prices = prices
        self.features = features
        self.seq_len = seq_len
        self.forecast_horizons = forecast_horizons
        
        # Maximum horizon to ensure we have targets
        self.max_horizon = max(forecast_horizons)
    
    def __len__(self):
        return len(self.prices) - self.seq_len - self.max_horizon
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.features[idx:idx + self.seq_len]
        
        # Targets for each horizon
        targets = {}
        for h in self.forecast_horizons:
            targets[h] = self.prices[idx + self.seq_len + h - 1]
        
        return torch.FloatTensor(x), targets


def train_transformer_model(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> PriceForecaster:
    """
    Train transformer forecasting model.
    
    Args:
        train_data: (prices, features) for training
        val_data: (prices, features) for validation
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Create datasets
    train_prices, train_features = train_data
    val_prices, val_features = val_data
    
    train_dataset = PriceDataset(train_prices, train_features)
    val_dataset = PriceDataset(val_prices, val_features)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create model
    model = PriceForecaster(
        input_features=train_features.shape[1],
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_targets in train_loader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            forecasts = model(batch_x)
            
            # Calculate loss (negative log-likelihood assuming Gaussian)
            loss = 0.0
            for horizon in model.forecast_horizons:
                target = batch_targets[horizon].to(device)
                mean = forecasts[horizon]["mean"]
                std = forecasts[horizon]["std"]
                
                # Negative log-likelihood
                nll = 0.5 * torch.log(2 * np.pi * std**2) + \
                      0.5 * ((target - mean) ** 2) / (std ** 2)
                loss += nll.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_targets in val_loader:
                batch_x = batch_x.to(device)
                
                forecasts = model(batch_x)
                
                loss = 0.0
                for horizon in model.forecast_horizons:
                    target = batch_targets[horizon].to(device)
                    mean = forecasts[horizon]["mean"]
                    std = forecasts[horizon]["std"]
                    
                    nll = 0.5 * torch.log(2 * np.pi * std**2) + \
                          0.5 * ((target - mean) ** 2) / (std ** 2)
                    loss += nll.mean()
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer_model.pth")
    
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load("best_transformer_model.pth"))
    
    return model


if __name__ == "__main__":
    # Test transformer model
    logger.info("Testing Transformer Price Forecaster")
    
    # Generate synthetic data
    np.random.seed(42)
    seq_len = 1000
    
    # Prices with trend and seasonality
    t = np.arange(seq_len)
    prices = 50 + 0.01 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(seq_len) * 5
    
    # Features: hour, day_of_week, lagged prices, etc.
    features = np.column_stack([
        t % 24,  # Hour of day
        (t // 24) % 7,  # Day of week
        np.roll(prices, 1),  # Lag-1 price
        np.roll(prices, 24),  # Lag-24 price
        np.roll(prices, 168),  # Lag-168 price (1 week)
    ])
    
    # Split data
    split = int(0.8 * seq_len)
    train_data = (prices[:split], features[:split])
    val_data = (prices[split:], features[split:])
    
    # Train model
    model = train_transformer_model(
        train_data,
        val_data,
        epochs=10,
        batch_size=16,
    )
    
    logger.info("Transformer model training complete!")
