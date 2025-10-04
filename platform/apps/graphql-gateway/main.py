"""
GraphQL Gateway
Flexible querying with reduced overfetching and improved developer experience.
"""
import logging
from datetime import datetime, date
from typing import List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
import asyncpg
from clickhouse_driver import Client

from infrastructure_schema import InfrastructureQuery
from analytics_schema import AnalyticsQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GraphQL Types

@strawberry.type
class Instrument:
    instrument_id: str
    market: str
    product: str
    location_code: str
    timezone: str
    unit: str
    currency: str


@strawberry.type
class PriceTick:
    event_time: datetime
    instrument_id: str
    location_code: str
    price_type: str
    value: float
    volume: Optional[float]
    currency: str
    unit: str
    source: str


@strawberry.type
class CurvePoint:
    delivery_start: date
    delivery_end: date
    tenor_type: str
    price: float
    currency: str
    unit: str


@strawberry.type
class ForwardCurve:
    instrument_id: str
    as_of_date: date
    scenario_id: str
    points: List[CurvePoint]


@strawberry.type
class Scenario:
    scenario_id: str
    title: str
    description: str
    visibility: str
    created_by: str
    created_at: datetime


@strawberry.type
class BacktestResult:
    instrument_id: str
    scenario_id: str
    forecast_date: date
    mape: float
    wape: float
    rmse: float
    mean_error: float
    n_observations: int


# GraphQL Queries

@strawberry.type
class Query(AnalyticsQuery, InfrastructureQuery):
    
    @strawberry.field
    async def instruments(
        self,
        market: Optional[str] = None,
        product: Optional[str] = None,
        limit: int = 100,
    ) -> List[Instrument]:
        """Get instruments with optional filtering."""
        pool = await get_pg_pool()
        
        query = "SELECT * FROM pg.instrument WHERE 1=1"
        params = []
        
        if market:
            query += f" AND market = ${len(params) + 1}"
            params.append(market)
        if product:
            query += f" AND product = ${len(params) + 1}"
            params.append(product)
        
        query += f" LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            Instrument(
                instrument_id=r["instrument_id"],
                market=r["market"],
                product=r["product"],
                location_code=r["location_code"],
                timezone=r["timezone"],
                unit=r["unit"],
                currency=r["currency"],
            )
            for r in rows
        ]
    
    @strawberry.field
    async def price_ticks(
        self,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        price_type: str = "mid",
        limit: int = 1000,
    ) -> List[PriceTick]:
        """Get historical price ticks."""
        ch_client = get_ch_client()
        
        query = """
        SELECT 
            event_time,
            instrument_id,
            location_code,
            price_type,
            value,
            volume,
            currency,
            unit,
            source
        FROM ch.market_price_ticks
        WHERE instrument_id = %(id)s
          AND event_time BETWEEN %(start)s AND %(end)s
          AND price_type = %(price_type)s
        ORDER BY event_time DESC
        LIMIT %(limit)s
        """
        
        result = ch_client.execute(
            query,
            {
                "id": instrument_id,
                "start": start_time,
                "end": end_time,
                "price_type": price_type,
                "limit": limit,
            },
        )
        
        return [
            PriceTick(
                event_time=row[0],
                instrument_id=row[1],
                location_code=row[2],
                price_type=row[3],
                value=row[4],
                volume=row[5],
                currency=row[6],
                unit=row[7],
                source=row[8],
            )
            for row in result
        ]
    
    @strawberry.field
    async def forward_curve(
        self,
        instrument_id: str,
        as_of_date: date,
        scenario_id: str = "BASE",
    ) -> ForwardCurve:
        """Get forward curve for instrument."""
        ch_client = get_ch_client()
        
        query = """
        SELECT 
            delivery_start,
            delivery_end,
            tenor_type,
            price,
            currency,
            unit
        FROM ch.forward_curve_points
        WHERE instrument_id = %(id)s
          AND as_of_date = %(date)s
          AND scenario_id = %(scenario)s
        ORDER BY delivery_start
        """
        
        result = ch_client.execute(
            query,
            {
                "id": instrument_id,
                "date": as_of_date,
                "scenario": scenario_id,
            },
        )
        
        points = [
            CurvePoint(
                delivery_start=row[0],
                delivery_end=row[1],
                tenor_type=row[2],
                price=row[3],
                currency=row[4],
                unit=row[5],
            )
            for row in result
        ]
        
        return ForwardCurve(
            instrument_id=instrument_id,
            as_of_date=as_of_date,
            scenario_id=scenario_id,
            points=points,
        )
    
    @strawberry.field
    async def scenarios(
        self,
        visibility: Optional[str] = None,
        limit: int = 50,
    ) -> List[Scenario]:
        """Get available scenarios."""
        pool = await get_pg_pool()
        
        query = "SELECT * FROM pg.scenario WHERE 1=1"
        params = []
        
        if visibility:
            query += f" AND visibility = ${len(params) + 1}"
            params.append(visibility)
        
        query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            Scenario(
                scenario_id=r["scenario_id"],
                title=r["title"],
                description=r["description"],
                visibility=r["visibility"],
                created_by=r["created_by"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
    
    @strawberry.field
    async def backtest_results(
        self,
        instrument_id: str,
        scenario_id: str = "BASE",
        limit: int = 10,
    ) -> List[BacktestResult]:
        """Get backtest results for instrument."""
        pool = await get_pg_pool()
        
        query = """
        SELECT 
            instrument_id,
            scenario_id,
            forecast_date,
            mape,
            wape,
            rmse,
            mean_error,
            n_observations
        FROM pg.backtest_results
        WHERE instrument_id = $1
          AND scenario_id = $2
        ORDER BY forecast_date DESC
        LIMIT $3
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, instrument_id, scenario_id, limit)
        
        return [
            BacktestResult(
                instrument_id=r["instrument_id"],
                scenario_id=r["scenario_id"],
                forecast_date=r["forecast_date"],
                mape=r["mape"],
                wape=r["wape"],
                rmse=r["rmse"],
                mean_error=r["mean_error"],
                n_observations=r["n_observations"],
            )
            for r in rows
        ]


# Database connections
_pg_pool = None
_ch_client = None


async def get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(
            "postgresql://postgres:postgres@postgres:5432/market_intelligence"
        )
    return _pg_pool


def get_ch_client():
    global _ch_client
    if _ch_client is None:
        _ch_client = Client(host="clickhouse", port=9000)
    return _ch_client


# Create GraphQL schema
schema = strawberry.Schema(query=Query)

# Create FastAPI app
app = FastAPI(title="254Carbon GraphQL Gateway")

# Add GraphQL router with context
async def get_context():
    return {
        "pg_pool": await get_pg_pool(),
        "ch_client": get_ch_client()
    }

graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
