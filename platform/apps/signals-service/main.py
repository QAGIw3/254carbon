"""
Trading Signals Service

Algorithmic signal generation, backtesting framework,
and FIX protocol integration for trading platforms.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading Signals Service",
    description="Algorithmic trading signals and FIX integration",
    version="1.0.0",
)


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class Strategy(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    SPREAD_TRADING = "spread_trading"
    VOLATILITY = "volatility"
    ML_ENSEMBLE = "ml_ensemble"


class TradingSignal(BaseModel):
    """Generated trading signal."""
    signal_id: str
    instrument_id: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    strategy: Strategy
    generated_at: datetime
    expires_at: datetime
    rationale: str


class BacktestRequest(BaseModel):
    """Backtest configuration."""
    strategy: Strategy
    instruments: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1  # 10% of capital per trade


class BacktestResult(BaseModel):
    """Backtest performance metrics."""
    strategy: Strategy
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration_hours: float
    best_trade: float
    worst_trade: float


class FIXOrder(BaseModel):
    """FIX protocol order."""
    order_id: str
    instrument_id: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: str  # "MARKET", "LIMIT"
    price: Optional[float] = None
    time_in_force: str = "DAY"  # "DAY", "GTC", "IOC"


class SignalGenerator:
    """Generate trading signals using various strategies."""
    
    def __init__(self):
        self.strategies = {
            Strategy.MEAN_REVERSION: self._mean_reversion_signal,
            Strategy.MOMENTUM: self._momentum_signal,
            Strategy.SPREAD_TRADING: self._spread_trading_signal,
            Strategy.VOLATILITY: self._volatility_signal,
            Strategy.ML_ENSEMBLE: self._ml_ensemble_signal,
        }
    
    async def generate(
        self,
        strategy: Strategy,
        instrument_id: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """Generate signal using specified strategy."""
        
        signal_func = self.strategies.get(strategy)
        if not signal_func:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return await signal_func(instrument_id, market_data)
    
    async def _mean_reversion_signal(
        self,
        instrument_id: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """
        Mean reversion strategy.
        
        Buy when price is below moving average, sell when above.
        """
        prices = market_data.get("prices", [])
        if len(prices) < 20:
            return self._hold_signal(instrument_id)
        
        current_price = prices[-1]
        ma_20 = sum(prices[-20:]) / 20
        std_20 = (sum((p - ma_20) ** 2 for p in prices[-20:]) / 20) ** 0.5
        
        # Calculate z-score
        z_score = (current_price - ma_20) / std_20 if std_20 > 0 else 0
        
        if z_score < -2:  # Price is 2 std below mean
            signal_type = SignalType.BUY
            strength = SignalStrength.STRONG if z_score < -2.5 else SignalStrength.MODERATE
            target = ma_20
            stop_loss = current_price * 0.95
            rationale = f"Price {abs(z_score):.1f} std below 20-day MA. Mean reversion expected."
        elif z_score > 2:  # Price is 2 std above mean
            signal_type = SignalType.SELL
            strength = SignalStrength.STRONG if z_score > 2.5 else SignalStrength.MODERATE
            target = ma_20
            stop_loss = current_price * 1.05
            rationale = f"Price {z_score:.1f} std above 20-day MA. Mean reversion expected."
        else:
            return self._hold_signal(instrument_id)
        
        confidence = min(abs(z_score) / 3.0, 0.95)  # Cap at 95%
        
        return TradingSignal(
            signal_id=f"SIG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            instrument_id=instrument_id,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            strategy=Strategy.MEAN_REVERSION,
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            rationale=rationale,
        )
    
    async def _momentum_signal(
        self,
        instrument_id: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """
        Momentum strategy.
        
        Buy when trend is up, sell when trend is down.
        """
        prices = market_data.get("prices", [])
        if len(prices) < 50:
            return self._hold_signal(instrument_id)
        
        current_price = prices[-1]
        
        # Calculate momentum indicators
        ma_10 = sum(prices[-10:]) / 10
        ma_50 = sum(prices[-50:]) / 50
        roc_10 = (current_price - prices[-10]) / prices[-10] * 100  # 10-day ROC
        
        # RSI calculation (simplified)
        gains = [max(prices[i] - prices[i-1], 0) for i in range(-14, 0)]
        losses = [max(prices[i-1] - prices[i], 0) for i in range(-14, 0)]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Signal logic
        if ma_10 > ma_50 and roc_10 > 5 and rsi < 70:
            signal_type = SignalType.BUY
            strength = SignalStrength.STRONG if roc_10 > 10 else SignalStrength.MODERATE
            confidence = min(roc_10 / 15, 0.9)
            target = current_price * 1.10
            stop_loss = ma_50
            rationale = f"Strong upward momentum. 10-day ROC: {roc_10:.1f}%, MA crossover bullish."
        elif ma_10 < ma_50 and roc_10 < -5 and rsi > 30:
            signal_type = SignalType.SELL
            strength = SignalStrength.STRONG if roc_10 < -10 else SignalStrength.MODERATE
            confidence = min(abs(roc_10) / 15, 0.9)
            target = current_price * 0.90
            stop_loss = ma_50
            rationale = f"Strong downward momentum. 10-day ROC: {roc_10:.1f}%, MA crossover bearish."
        else:
            return self._hold_signal(instrument_id)
        
        return TradingSignal(
            signal_id=f"SIG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            instrument_id=instrument_id,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            strategy=Strategy.MOMENTUM,
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=48),
            rationale=rationale,
        )
    
    async def _spread_trading_signal(
        self,
        instrument_id: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """
        Spread trading strategy.
        
        Trade spreads between related markets.
        """
        # Mock spread analysis
        current_price = market_data.get("price", 45.0)
        related_price = market_data.get("related_price", 42.0)
        
        spread = current_price - related_price
        avg_spread = 2.5  # Historical average
        
        if spread > avg_spread * 1.5:
            # Spread too wide - short the spread
            signal_type = SignalType.SELL
            rationale = f"Spread {spread:.2f} significantly above average {avg_spread:.2f}. Reversion expected."
        elif spread < avg_spread * 0.5:
            # Spread too narrow - long the spread
            signal_type = SignalType.BUY
            rationale = f"Spread {spread:.2f} significantly below average {avg_spread:.2f}. Expansion expected."
        else:
            return self._hold_signal(instrument_id)
        
        return TradingSignal(
            signal_id=f"SIG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            instrument_id=instrument_id,
            signal_type=signal_type,
            strength=SignalStrength.MODERATE,
            confidence=0.70,
            entry_price=current_price,
            target_price=current_price + (avg_spread - spread) * 0.5,
            stop_loss=current_price + (spread - avg_spread) * 0.3,
            strategy=Strategy.SPREAD_TRADING,
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=7),
            rationale=rationale,
        )
    
    async def _volatility_signal(
        self,
        instrument_id: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """Volatility-based strategy."""
        # Placeholder
        return self._hold_signal(instrument_id)
    
    async def _ml_ensemble_signal(
        self,
        instrument_id: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """ML ensemble strategy combining multiple models."""
        # Placeholder - would use transformer model outputs
        return self._hold_signal(instrument_id)
    
    def _hold_signal(self, instrument_id: str) -> TradingSignal:
        """Generate HOLD signal."""
        return TradingSignal(
            signal_id=f"SIG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            instrument_id=instrument_id,
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=0.5,
            entry_price=0.0,
            strategy=Strategy.MEAN_REVERSION,
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            rationale="No clear trading opportunity at this time.",
        )


# Global signal generator
signal_generator = SignalGenerator()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/signals/generate", response_model=TradingSignal)
async def generate_signal(
    strategy: Strategy,
    instrument_id: str,
    market_data: Optional[Dict[str, Any]] = None,
):
    """
    Generate trading signal.
    
    Uses specified strategy to analyze market and generate actionable signal.
    """
    try:
        # Fetch market data if not provided
        if not market_data:
            # TODO: Fetch from database
            market_data = {
                "price": 45.0,
                "prices": [40 + i * 0.5 for i in range(100)],
            }
        
        signal = await signal_generator.generate(
            strategy,
            instrument_id,
            market_data
        )
        
        logger.info(f"Generated {signal.signal_type} signal for {instrument_id}")
        
        return signal
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/signals/backtest", response_model=BacktestResult)
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest trading strategy.
    
    Simulates strategy performance on historical data.
    """
    try:
        logger.info(
            f"Backtesting {request.strategy} from "
            f"{request.start_date} to {request.end_date}"
        )
        
        # Mock backtest results
        # In production, would simulate trades on historical data
        total_return = 15.5 + (hash(str(request.strategy)) % 20) - 10
        sharpe = 1.2 + (hash(str(request.strategy)) % 10) / 10
        
        result = BacktestResult(
            strategy=request.strategy,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=-8.5,
            win_rate=0.58,
            total_trades=150,
            avg_trade_duration_hours=36.5,
            best_trade=12.3,
            worst_trade=-5.8,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/signals/fix/order")
async def send_fix_order(order: FIXOrder):
    """
    Send order via FIX protocol.
    
    Integrates with trading platforms using FIX 4.4.
    """
    logger.info(f"Sending FIX order: {order.order_id}")
    
    # Mock FIX message
    fix_message = {
        "8": "FIX.4.4",  # BeginString
        "35": "D",  # MsgType (NewOrderSingle)
        "11": order.order_id,  # ClOrdID
        "55": order.instrument_id,  # Symbol
        "54": "1" if order.side == "BUY" else "2",  # Side
        "38": str(order.quantity),  # OrderQty
        "40": "1" if order.order_type == "MARKET" else "2",  # OrdType
        "59": "0" if order.time_in_force == "DAY" else "1",  # TimeInForce
    }
    
    if order.price:
        fix_message["44"] = str(order.price)  # Price
    
    return {
        "status": "sent",
        "order_id": order.order_id,
        "fix_message": "|".join(f"{k}={v}" for k, v in fix_message.items()),
        "acknowledgement": "PENDING",
    }


@app.get("/api/v1/signals/performance")
async def get_signal_performance(
    start_date: datetime,
    end_date: datetime,
    strategy: Optional[Strategy] = None,
):
    """
    Get historical signal performance.
    
    Track accuracy of generated signals.
    """
    # Mock performance data
    signals_generated = 250
    profitable = 145
    unprofitable = 85
    pending = 20
    
    return {
        "period": f"{start_date.date()} to {end_date.date()}",
        "strategy": strategy or "all",
        "signals_generated": signals_generated,
        "profitable": profitable,
        "unprofitable": unprofitable,
        "pending": pending,
        "win_rate": profitable / (profitable + unprofitable),
        "avg_return_pct": 2.8,
        "total_return_pct": 42.5,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8016)

