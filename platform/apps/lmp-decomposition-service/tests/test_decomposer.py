import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date

import pytest

# Ensure local service directory is importable
SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from decomposer import LMPDecomposer
from ptdf_calculator import PTDFCalculator


@pytest.mark.asyncio
async def test_calculate_loss_component_distance_factor():
    decomposer = LMPDecomposer()

    energy = 50.0
    # Without distance (heuristic path)
    loss1 = decomposer.calculate_loss_component("PJM.NODE.A", energy, "PJM")
    # With explicit distance increases loss
    loss2 = decomposer.calculate_loss_component("PJM.NODE.A", energy, "PJM", distance_from_hub=50.0)

    assert loss1 > 0
    assert loss2 > loss1


@pytest.mark.asyncio
async def test_calculate_electrical_distance_and_ptdf():
    decomposer = LMPDecomposer()
    ptdf = PTDFCalculator()

    network = await ptdf.get_network_topology("PJM")
    # Known nodes in mock network
    distance = await decomposer.calculate_electrical_distance("PJM.BUS2", "PJM", network)

    assert distance >= 0.0

    # PTDF should be bounded and finite
    value = ptdf.calculate_ptdf("PJM.BUS1", "PJM.BUS2", "PJM.BUS1_PJM.BUS2", network)
    assert -1.0 <= value <= 1.0


@pytest.mark.asyncio
async def test_get_lmp_data_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "1")
    decomposer = LMPDecomposer()
    start = datetime.utcnow() - timedelta(hours=4)
    end = datetime.utcnow()
    df = await decomposer.get_lmp_data(["PJM.NODE.1"], start, end, "PJM")
    assert not df.empty
    assert set(["timestamp", "node_id", "lmp"]).issubset(df.columns)


