# Multimodal Transformer Runbook

Operational checklist for training, validating, and serving the multimodal transformer architecture.

## Prerequisites
- Multimodal mapping configured in `platform/apps/ml-service/config/multimodal_mapping.yaml`
- ML service dependencies installed (`requirements.txt`)
- Access to historical price, fundamentals, and weather data sources

## Training Workflow
1. **Select instruments**: choose instrument IDs covering interconnected commodities (e.g., power hub + gas hub).
2. **Launch training**:
   ```bash
   curl -X POST http://ml-service:8006/api/v1/ml/train \
     -H "Content-Type: application/json" \
     -d '{
           "instrument_ids": ["POWER.NYISO.ZONEA", "GAS.HENRY_HUB.MONTH_AHEAD"],
           "start_date": "2024-01-01",
           "end_date": "2024-12-31",
           "model_type": "multimodal_transformer",
           "hyperparameters": {
             "seq_len": 128,
             "forecast_horizons": [7, 30, 90],
             "epochs": 40,
             "learning_rate": 0.0004
           }
         }'
   ```
3. **Monitor MLflow run**: metrics logged under experiment `multimodal_forecast_<joined_instruments>`.
4. **Registry check**: confirm a new version exists for each instrument via `/api/v1/ml/models/<instrument_id>`.

## Forecast Execution
```bash
curl -X POST http://ml-service:8006/api/v1/ml/forecast \
  -H "Content-Type: application/json" \
  -d '{"instrument_id": "POWER.NYISO.ZONEA", "model_version": null}'
```
`extras.fusion_gates` and `extras.cross_attention` provide interpretability for modality usage and inter-market influence.

## Troubleshooting
| Symptom | Checks | Remediation |
|---------|--------|-------------|
| 400 "Insufficient data" | Verify lookbacks in `multimodal_mapping` cover requested window | Increase lookback days or adjust `start_date` |
| Training diverges | Inspect MLflow loss curve | Reduce learning rate, increase `seq_len`, enable regularization (`weight_decay`) |
| Missing commodities in output | Ensure each instrument has entry in YAML mapping | Update config and rebuild feature bundle |

## Operational Notes
- Head gating masks derive from `defaults.commodity_groups.*.fused_heads`; update to restrict attention heads per market cluster.
- `artifacts.fusion_snapshot` stored in metadata can be surfaced in dashboards.
- For rapid backtesting, lower `seq_len` and horizons; retrain with production values once validated.
