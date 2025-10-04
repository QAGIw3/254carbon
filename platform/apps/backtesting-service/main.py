"""
Backtesting Service
Compares forecasts against realized prices and computes accuracy metrics.
"""
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from db import get_clickhouse_client, get_postgres_pool
from metrics import calculate_mape, calculate_wape, calculate_rmse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Backtesting Service",
    description="Forecast accuracy evaluation and metrics",
    version="1.0.0",
)


class BacktestRequest(BaseModel):
    instrument_id: str
    scenario_id: str
    forecast_date: date
    evaluation_start: date
    evaluation_end: date
    tenor_type: str = "Month"


class BacktestResult(BaseModel):
    instrument_id: str
    scenario_id: str
    forecast_date: date
    evaluation_period_days: int
    mape: float
    wape: float
    rmse: float
    mean_error: float
    median_error: float
    n_observations: int


class BacktestSummary(BaseModel):
    market: str
    product: str
    period: str
    results: List[BacktestResult]
    aggregate_mape: float
    aggregate_wape: float
    aggregate_rmse: float


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/backtest/run", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """
    Run backtest for a single instrument and forecast.
    
    Compares forward curve forecasts against realized prices.
    """
    logger.info(
        f"Running backtest for {request.instrument_id}, "
        f"forecast_date={request.forecast_date}"
    )
    
    try:
        ch_client = get_clickhouse_client()
        
        # Get forecast values
        forecast_query = """
        SELECT 
            delivery_start,
            delivery_end,
            price as forecast_price
        FROM market_intelligence.forward_curve_points
        WHERE instrument_id = %(instrument_id)s
          AND scenario_id = %(scenario_id)s
          AND as_of_date = %(forecast_date)s
          AND tenor_type = %(tenor_type)s
          AND delivery_start >= %(eval_start)s
          AND delivery_start <= %(eval_end)s
        ORDER BY delivery_start
        """
        
        forecasts = ch_client.execute(
            forecast_query,
            {
                'instrument_id': request.instrument_id,
                'scenario_id': request.scenario_id,
                'forecast_date': request.forecast_date,
                'tenor_type': request.tenor_type,
                'eval_start': request.evaluation_start,
                'eval_end': request.evaluation_end,
            },
        )
        
        if not forecasts:
            raise HTTPException(
                status_code=404,
                detail="No forecast data found for specified parameters"
            )
        
        # Get realized values
        realized_query = """
        SELECT 
            toDate(event_time) as date,
            avg(value) as realized_price
        FROM market_intelligence.market_price_ticks
        WHERE instrument_id = %(instrument_id)s
          AND price_type = 'settle'
          AND toDate(event_time) >= %(eval_start)s
          AND toDate(event_time) <= %(eval_end)s
        GROUP BY date
        ORDER BY date
        """
        
        realized = ch_client.execute(
            realized_query,
            {
                'instrument_id': request.instrument_id,
                'eval_start': request.evaluation_start,
                'eval_end': request.evaluation_end,
            },
        )
        
        if not realized:
            raise HTTPException(
                status_code=404,
                detail="No realized data found for evaluation period"
            )
        
        # Match forecasts to realized values
        forecast_df = pd.DataFrame(
            forecasts,
            columns=['delivery_start', 'delivery_end', 'forecast_price']
        )
        realized_df = pd.DataFrame(
            realized,
            columns=['date', 'realized_price']
        )
        
        # Aggregate realized to monthly if needed
        if request.tenor_type == "Month":
            realized_df['month'] = pd.to_datetime(realized_df['date']).dt.to_period('M')
            realized_monthly = realized_df.groupby('month')['realized_price'].mean().reset_index()
            realized_monthly['date'] = realized_monthly['month'].dt.to_timestamp()
        else:
            realized_monthly = realized_df
        
        # Join forecasts with realized
        merged = pd.merge(
            forecast_df,
            realized_monthly,
            left_on='delivery_start',
            right_on='date',
            how='inner'
        )
        
        if merged.empty:
            raise HTTPException(
                status_code=400,
                detail="No matching data points between forecast and realized"
            )
        
        # Calculate metrics
        forecast_values = merged['forecast_price'].values
        realized_values = merged['realized_price'].values
        
        mape = calculate_mape(realized_values, forecast_values)
        wape = calculate_wape(realized_values, forecast_values)
        rmse = calculate_rmse(realized_values, forecast_values)
        
        errors = forecast_values - realized_values
        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))
        
        # Store results in PostgreSQL
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pg.backtest_results 
                (instrument_id, scenario_id, forecast_date, evaluation_start, 
                 evaluation_end, mape, wape, rmse, mean_error, median_error, 
                 n_observations, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
                """,
                request.instrument_id,
                request.scenario_id,
                request.forecast_date,
                request.evaluation_start,
                request.evaluation_end,
                mape,
                wape,
                rmse,
                mean_error,
                median_error,
                len(merged),
            )
        
        evaluation_days = (request.evaluation_end - request.evaluation_start).days
        
        return BacktestResult(
            instrument_id=request.instrument_id,
            scenario_id=request.scenario_id,
            forecast_date=request.forecast_date,
            evaluation_period_days=evaluation_days,
            mape=mape,
            wape=wape,
            rmse=rmse,
            mean_error=mean_error,
            median_error=median_error,
            n_observations=len(merged),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/backtest/summary/{market}/{product}")
async def get_backtest_summary(
    market: str,
    product: str,
    period: str = "last_30_days",
) -> BacktestSummary:
    """Get aggregate backtest metrics for market/product."""
    
    # Parse period
    if period == "last_30_days":
        start_date = datetime.utcnow() - timedelta(days=30)
    elif period == "last_90_days":
        start_date = datetime.utcnow() - timedelta(days=90)
    else:
        start_date = datetime.utcnow() - timedelta(days=30)
    
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT 
                    br.instrument_id,
                    br.scenario_id,
                    br.forecast_date,
                    br.evaluation_end - br.evaluation_start as evaluation_period_days,
                    br.mape,
                    br.wape,
                    br.rmse,
                    br.mean_error,
                    br.median_error,
                    br.n_observations
                FROM pg.backtest_results br
                JOIN pg.instrument i ON br.instrument_id = i.instrument_id
                WHERE i.market = $1
                  AND i.product = $2
                  AND br.created_at >= $3
                ORDER BY br.forecast_date DESC
                """,
                market,
                product,
                start_date,
            )
        
        backtest_results = [
            BacktestResult(
                instrument_id=r['instrument_id'],
                scenario_id=r['scenario_id'],
                forecast_date=r['forecast_date'],
                evaluation_period_days=r['evaluation_period_days'],
                mape=r['mape'],
                wape=r['wape'],
                rmse=r['rmse'],
                mean_error=r['mean_error'],
                median_error=r['median_error'],
                n_observations=r['n_observations'],
            )
            for r in results
        ]
        
        # Calculate aggregates
        if backtest_results:
            aggregate_mape = np.mean([r.mape for r in backtest_results])
            aggregate_wape = np.mean([r.wape for r in backtest_results])
            aggregate_rmse = np.mean([r.rmse for r in backtest_results])
        else:
            aggregate_mape = aggregate_wape = aggregate_rmse = 0.0
        
        return BacktestSummary(
            market=market,
            product=product,
            period=period,
            results=backtest_results,
            aggregate_mape=aggregate_mape,
            aggregate_wape=aggregate_wape,
            aggregate_rmse=aggregate_rmse,
        )
        
    except Exception as e:
        logger.error(f"Error fetching backtest summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

