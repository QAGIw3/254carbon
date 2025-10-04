import importlib

import numpy as np
import pandas as pd
import pytest

_torch_spec = importlib.util.find_spec("torch")


if _torch_spec is None:

    def test_multimodal_transformer_requires_torch():
        pytest.skip("torch not installed")

else:
    import torch  # type: ignore[no-redef]

    from deep_learning import MultiCommodityTransformer
    from multimodal_dataset import MultiModalDataset

    def _build_feature_bundle(num_points: int = 16):
        index = pd.date_range("2023-01-01", periods=num_points, freq="D")

        def _mask(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(index=index)
            return pd.DataFrame(np.ones_like(df.values, dtype=float), index=index, columns=df.columns)

        price_a = pd.DataFrame({"price::A": np.linspace(1.0, 2.0, num_points)}, index=index)
        fund_a = pd.DataFrame(
            {
                "fund::load": np.linspace(0.1, 0.9, num_points),
                "fund::gen": np.linspace(0.2, 1.2, num_points),
            },
            index=index,
        )

        price_b = pd.DataFrame({"price::B": np.linspace(1.5, 2.5, num_points)}, index=index)
        fund_b = pd.DataFrame({"fund::storage": np.linspace(0.3, 1.3, num_points)}, index=index)
        weather_b = pd.DataFrame({"wx::temp": np.linspace(-5.0, 5.0, num_points)}, index=index)

        bundle = {
            "commodity_a": {
                "instrument_id": "INSTR.A",
                "commodity_group": "power",
                "modalities": {
                    "price": price_a,
                    "fundamentals": fund_a,
                    "weather": pd.DataFrame(index=index),
                },
                "masks": {
                    "price": _mask(price_a),
                    "fundamentals": _mask(fund_a),
                    "weather": pd.DataFrame(index=index),
                },
                "time_index": index,
            },
            "commodity_b": {
                "instrument_id": "INSTR.B",
                "commodity_group": "gas",
                "modalities": {
                    "price": price_b,
                    "fundamentals": fund_b,
                    "weather": weather_b,
                },
                "masks": {
                    "price": _mask(price_b),
                    "fundamentals": _mask(fund_b),
                    "weather": _mask(weather_b),
                },
                "time_index": index,
            },
        }
        return bundle

    def test_multimodal_dataset_shapes():
        bundle = _build_feature_bundle()
        dataset = MultiModalDataset(bundle, seq_len=5, forecast_horizons=[1, 2])

        sample = dataset[0]
        assert "commodity_a" in sample.targets
        assert sample.targets["commodity_a"].shape == (2,)

        batch = MultiModalDataset.collate_fn([sample])
        price_tensor = batch["commodities"]["commodity_a"]["modalities"]["price"]
        assert price_tensor.shape == (1, 5, 1)

        fund_tensor = batch["commodities"]["commodity_a"]["modalities"]["fundamentals"]
        assert fund_tensor.shape == (1, 5, 2)

        weather_tensor = batch["commodities"]["commodity_a"]["modalities"]["weather"]
        assert weather_tensor.shape == (1, 5, 0)

    def test_multicommodity_transformer_forward_pass():
        torch.manual_seed(42)
        bundle = _build_feature_bundle(num_points=12)

        commodity_modalities = {
            commodity: {
                modality: frame.shape[1]
                for modality, frame in payload["modalities"].items()
            }
            for commodity, payload in bundle.items()
        }
        commodity_groups = {
            commodity: payload["commodity_group"]
            for commodity, payload in bundle.items()
        }

        model = MultiCommodityTransformer(
            commodity_modalities=commodity_modalities,
            commodity_groups=commodity_groups,
            d_model=32,
            num_heads=4,
            num_layers=1,
            cross_commodity_layers=1,
            d_ff=64,
            dropout=0.1,
            forecast_horizons=[1, 2],
            head_mask_config={"power": [0, 1], "gas": [1, 2], "commodity_b": [1, 2]},
        )

        batch_size = 3
        seq_len = 6
        commodity_inputs = {}
        commodity_masks = {}

        for commodity, modalities in commodity_modalities.items():
            modality_inputs = {}
            modality_masks = {}
            for modality, feature_count in modalities.items():
                if feature_count == 0:
                    modality_inputs[modality] = torch.zeros(batch_size, seq_len, 0)
                    modality_masks[modality] = torch.zeros(batch_size, seq_len, 0)
                else:
                    modality_inputs[modality] = torch.randn(batch_size, seq_len, feature_count)
                    modality_masks[modality] = torch.ones(batch_size, seq_len, feature_count)
            commodity_inputs[commodity] = modality_inputs
            commodity_masks[commodity] = modality_masks

        outputs = model(
            commodity_inputs,
            commodity_masks=commodity_masks,
            return_attentions=True,
        )

        assert set(outputs["forecasts"].keys()) == {"commodity_a", "commodity_b"}
        for commodity, horizons in outputs["forecasts"].items():
            for horizon, stats in horizons.items():
                assert stats["mean"].shape == (batch_size,)
                assert stats["std"].shape == (batch_size,)
            assert set(outputs["fusion_gates"][commodity].keys()) == set(commodity_modalities[commodity].keys())

        attn_stack = outputs.get("cross_attentions")
        if attn_stack:
            last_layer = attn_stack[-1]
            assert last_layer.shape[1] == len(commodity_modalities)
