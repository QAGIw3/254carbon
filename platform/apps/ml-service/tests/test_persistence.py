from datetime import date

from refining_persistence import RefiningPersistence
from renewables_persistence import RenewablesPersistence


class FakeClient:
    def __init__(self):
        self.statements = []

    def execute(self, query, rows, types_check=False):  # noqa: D401 - signature matches ClickHouse
        self.statements.append((query, rows))

    def disconnect(self):
        pass


def test_refining_crack_persistence_builds_rows():
    client = FakeClient()
    persistence = RefiningPersistence(ch_client=client)
    records = [
        {
            "as_of_date": date(2024, 1, 1),
            "region": "PADD3",
            "refinery_id": "RF1",
            "crack_type": "3:2:1",
            "crude_code": "OIL.WTI",
            "gasoline_price": 2.75,
            "diesel_price": 3.05,
            "jet_price": 2.95,
            "crack_spread": 18.5,
            "margin_per_bbl": 6.2,
            "optimal_yields": {"gasoline": 0.5, "diesel": 0.3},
            "constraints": {"min_gasoline": 0.35},
            "diagnostics": {"note": "test"},
            "model_version": "vtest",
        }
    ]
    inserted = persistence.persist_crack_optimization(records)
    assert inserted == 1
    assert client.statements, "Expected ClickHouse insert to be called"
    query, rows = client.statements[0]
    assert "INSERT INTO ch.refining_crack_optimization" in query
    stored = rows[0]
    assert stored[0] == date(2024, 1, 1)
    assert stored[1] == "PADD3"
    assert stored[5] == 2.75
    assert stored[9] == 6.2


def test_renewables_rin_persistence_builds_rows():
    client = FakeClient()
    persistence = RenewablesPersistence(ch_client=client)
    records = [
        {
            "as_of_date": date(2024, 1, 1),
            "rin_category": "D4",
            "horizon_days": 30,
            "forecast_date": date(2024, 2, 1),
            "forecast_price": 1.25,
            "std": 0.15,
            "drivers": {"trend": "flat"},
            "model_version": "vtest",
        }
    ]
    inserted = persistence.persist_rin_forecast(records)
    assert inserted == 1
    assert client.statements, "Expected ClickHouse insert to be called"
    query, rows = client.statements[0]
    assert "INSERT INTO ch.rin_price_forecast" in query
    stored = rows[0]
    assert stored[0] == date(2024, 1, 1)
    assert stored[1] == "D4"
    assert stored[2] == 30
    assert stored[3] == date(2024, 2, 1)
    assert stored[4] == 1.25
