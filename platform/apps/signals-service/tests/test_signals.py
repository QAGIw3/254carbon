"""
Integration tests for Trading Signals Service.
"""
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


class TestTradingSignals:
    """Test suite for trading signals functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test client."""
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_generate_mean_reversion_signal(self):
        """Test mean reversion signal generation."""
        request_data = {
            "strategy": "mean_reversion",
            "instrument_id": "PJM.HUB.WEST",
            "market_data": {
                "price": 45.0,
                "prices": [40 + i * 0.5 for i in range(50)]  # Trending upward
            }
        }

        response = self.client.post("/api/v1/signals/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "signal_id" in data
        assert "instrument_id" in data
        assert "signal_type" in data
        assert "strength" in data
        assert "confidence" in data
        assert "entry_price" in data
        assert "strategy" in data
        assert "rationale" in data

        # Validate signal type is valid
        assert data["signal_type"] in ["BUY", "SELL", "HOLD"]

        # Validate confidence is in range [0, 1]
        assert 0 <= data["confidence"] <= 1

        # Validate strength
        assert data["strength"] in ["weak", "moderate", "strong"]

    def test_generate_momentum_signal(self):
        """Test momentum signal generation."""
        request_data = {
            "strategy": "momentum",
            "instrument_id": "MISO.HUB.INDIANA",
            "market_data": {
                "price": 50.0,
                "prices": [45 - i * 0.3 for i in range(60)]  # Trending downward
            }
        }

        response = self.client.post("/api/v1/signals/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["strategy"] == "momentum"
        assert data["instrument_id"] == "MISO.HUB.INDIANA"

    def test_generate_spread_trading_signal(self):
        """Test spread trading signal generation."""
        request_data = {
            "strategy": "spread_trading",
            "instrument_id": "ERCOT.HUB.NORTH",
            "market_data": {
                "price": 55.0,
                "related_price": 42.0  # Wide spread
            }
        }

        response = self.client.post("/api/v1/signals/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["strategy"] == "spread_trading"

    def test_generate_volatility_signal(self):
        """Test volatility signal generation."""
        request_data = {
            "strategy": "volatility",
            "instrument_id": "CAISO.HUB.SP15",
            "market_data": {
                "price": 48.0,
                "prices": [45 + (i % 5 - 2) * 2 for i in range(40)]  # Volatile
            }
        }

        response = self.client.post("/api/v1/signals/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["strategy"] == "volatility"

    def test_generate_ml_ensemble_signal(self):
        """Test ML ensemble signal generation."""
        request_data = {
            "strategy": "ml_ensemble",
            "instrument_id": "NYISO.HUB.ZONE_A",
            "market_data": {
                "price": 52.0,
                "prices": [50 + i * 0.1 for i in range(60)]  # Gentle uptrend
            }
        }

        response = self.client.post("/api/v1/signals/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["strategy"] == "ml_ensemble"

        # Ensemble should have moderate to strong confidence
        assert data["confidence"] >= 0.5

    def test_backtest_strategy(self):
        """Test strategy backtesting."""
        request_data = {
            "strategy": "mean_reversion",
            "instruments": ["PJM.HUB.WEST", "MISO.HUB.INDIANA"],
            "start_date": (datetime.utcnow() - timedelta(days=90)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "initial_capital": 100000.0,
            "position_size_pct": 0.1
        }

        response = self.client.post("/api/v1/signals/backtest", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "strategy" in data
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "win_rate" in data
        assert "total_trades" in data

        # Validate metrics are reasonable
        assert data["win_rate"] >= 0 and data["win_rate"] <= 1
        assert data["max_drawdown"] < 0  # Drawdown is negative
        assert data["total_trades"] > 0

    def test_send_fix_order(self):
        """Test FIX protocol order submission."""
        order_data = {
            "order_id": "ORD-TEST-001",
            "instrument_id": "PJM.HUB.WEST",
            "side": "BUY",
            "quantity": 100.0,
            "order_type": "LIMIT",
            "price": 50.0,
            "time_in_force": "DAY"
        }

        response = self.client.post("/api/v1/signals/fix/order", json=order_data)
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "order_id" in data
        assert "fix_message" in data

        # Validate FIX message format
        assert "8=FIX.4.4" in data["fix_message"]
        assert data["order_id"] == order_data["order_id"]

    def test_signal_performance(self):
        """Test signal performance tracking."""
        params = {
            "start_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "strategy": "mean_reversion"
        }

        response = self.client.get("/api/v1/signals/performance", params=params)
        assert response.status_code == 200

        data = response.json()
        assert "period" in data
        assert "signals_generated" in data
        assert "profitable" in data
        assert "unprofitable" in data
        assert "win_rate" in data
        assert "avg_return_pct" in data

        # Validate win rate
        win_rate = data["win_rate"]
        assert 0 <= win_rate <= 1

    def test_signal_with_insufficient_data(self):
        """Test signal generation with insufficient historical data."""
        request_data = {
            "strategy": "mean_reversion",
            "instrument_id": "TEST.INSTRUMENT",
            "market_data": {
                "price": 45.0,
                "prices": [45.0, 46.0, 44.0]  # Only 3 data points
            }
        }

        response = self.client.post("/api/v1/signals/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Should return HOLD signal due to insufficient data
        assert data["signal_type"] == "HOLD"

    def test_backtest_multiple_strategies(self):
        """Test backtesting with different strategies."""
        strategies = ["mean_reversion", "momentum", "spread_trading"]

        for strategy in strategies:
            request_data = {
                "strategy": strategy,
                "instruments": ["PJM.HUB.WEST"],
                "start_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "end_date": datetime.utcnow().isoformat(),
                "initial_capital": 50000.0
            }

            response = self.client.post("/api/v1/signals/backtest", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["strategy"] == strategy
            assert "sharpe_ratio" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

