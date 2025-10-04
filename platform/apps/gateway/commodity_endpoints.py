"""
Commodity-Specific API Endpoints

New endpoints for multi-commodity energy data:
- Oil benchmarks and differentials
- Natural gas hub pricing
- Coal indices and assessments
- Refined products pricing
- Biofuels and RINs data
- Carbon market data
"""
import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from auth import verify_token, has_permission
from db import get_clickhouse_client
from entitlements import check_entitlement
from cache import create_cache_decorator

logger = logging.getLogger(__name__)

# Create router
commodity_router = APIRouter(
    prefix="/api/v1/commodities",
    tags=["commodities"],
    dependencies=[Depends(verify_token)]
)


class CommodityPriceResponse(BaseModel):
    commodity_code: str
    price: float
    timestamp: datetime
    source: str
    location: Optional[str] = None
    quality_spec: Optional[Dict[str, Any]] = None


class FuturesCurveResponse(BaseModel):
    commodity_code: str
    as_of_date: date
    contract_month: date
    settlement_price: float
    open_interest: int
    volume: int
    exchange: str


class BenchmarkComparisonResponse(BaseModel):
    benchmark: str
    comparison_prices: Dict[str, float]
    price_differentials: Dict[str, float]
    timestamp: datetime


class StorageArbitrageScheduleEntry(BaseModel):
    date: date
    action: str
    volume_mmbtu: float
    inventory_mmbtu: float
    price: float
    net_cash_flow: float


class StorageArbitrageResponse(BaseModel):
    as_of_date: date
    hub: str
    region: Optional[str]
    expected_storage_value: float
    breakeven_spread: Optional[float]
    schedule: List[StorageArbitrageScheduleEntry]
    cost_parameters: Dict[str, Any]
    constraint_summary: Dict[str, Any]
    diagnostics: Dict[str, Any]


class WeatherImpactResponse(BaseModel):
    as_of_date: date
    entity_id: str
    coef_type: str
    coefficient: float
    r2: Optional[float]
    window: str
    diagnostics: Dict[str, Any]


class CoalToGasSwitchResponse(BaseModel):
    as_of_date: date
    region: str
    coal_cost_mwh: float
    gas_cost_mwh: float
    co2_price: float
    breakeven_gas_price: float
    switch_share: float
    diagnostics: Dict[str, Any]


class GasBasisModelResponse(BaseModel):
    as_of_date: date
    hub: str
    predicted_basis: float
    actual_basis: Optional[float]
    method: str
    diagnostics: Dict[str, Any]
    feature_snapshot: Dict[str, Any]


@commodity_router.get("/oil/benchmarks", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=300)  # 5 minute cache
async def get_oil_benchmarks(
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    commodities: List[str] = Query(default=["WTI", "BRENT", "DUBAI"], description="Oil commodities to include")
):
    """Get oil benchmark prices for comparison."""
    try:
        # Check entitlements for oil data access
        await check_entitlement("oil_data_access")

        clickhouse = await get_clickhouse_client()

        # Query oil benchmark prices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location,
            commodity_type
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'oil'
          AND instrument_id IN %(commodities)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'commodities': tuple(commodities),
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No oil benchmark data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching oil benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/gas/hubs/{hub_id}/prices", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=60)  # 1 minute cache for real-time data
async def get_gas_hub_prices(
    hub_id: str,
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    price_type: str = Query(default="spot", description="Price type (spot, futures)")
):
    """Get natural gas prices for specific hub."""
    try:
        # Check entitlements for gas data access
        await check_entitlement("gas_data_access")

        clickhouse = await get_clickhouse_client()

        # Query gas hub prices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location,
            price_type
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'gas'
          AND instrument_id = %(hub_id)s
          AND price_type = %(price_type)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'hub_id': hub_id,
                'price_type': price_type,
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"No gas price data found for hub {hub_id}")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching gas hub prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/coal/indices", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=3600)  # 1 hour cache for daily indices
async def get_coal_indices(
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    indices: List[str] = Query(default=["API2", "API4", "NEWC"], description="Coal indices to include")
):
    """Get coal price indices."""
    try:
        # Check entitlements for coal data access
        await check_entitlement("coal_data_access")

        clickhouse = await get_clickhouse_client()

        # Query coal indices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'coal'
          AND instrument_id IN %(indices)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'indices': tuple(indices),
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No coal index data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching coal indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/analytics/gas/arbitrage", response_model=StorageArbitrageResponse)
@create_cache_decorator(ttl_seconds=300)
async def get_gas_storage_arbitrage(
    hub: str = Query(..., description="Storage hub identifier"),
    as_of: date = Query(..., description="As-of date"),
):
    """Get gas storage arbitrage analytics."""
    try:
        await check_entitlement("gas_data_access")
        clickhouse = await get_clickhouse_client()

        query = """
        SELECT
            as_of_date,
            hub,
            region,
            expected_storage_value,
            breakeven_spread,
            optimal_schedule,
            cost_parameters,
            constraint_summary,
            diagnostics
        FROM ch.gas_storage_arbitrage
        WHERE hub = %(hub)s
          AND as_of_date = %(as_of)s
        ORDER BY created_at DESC
        LIMIT 1
        """

        result = await clickhouse.fetch(
            query,
            parameters={"hub": hub.upper(), "as_of": as_of}
        )

        if not result:
            raise HTTPException(status_code=404, detail="No storage arbitrage analytics found")

        row = result[0]
        schedule_payload = json.loads(row.get("optimal_schedule", "[]"))
        schedule = [StorageArbitrageScheduleEntry(**entry) for entry in schedule_payload]

        return StorageArbitrageResponse(
            as_of_date=row["as_of_date"],
            hub=row["hub"],
            region=row.get("region"),
            expected_storage_value=row["expected_storage_value"],
            breakeven_spread=row.get("breakeven_spread"),
            schedule=schedule,
            cost_parameters=json.loads(row.get("cost_parameters", "{}")),
            constraint_summary=json.loads(row.get("constraint_summary", "{}")),
            diagnostics=json.loads(row.get("diagnostics", "{}")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching gas storage arbitrage analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/analytics/weather-impact", response_model=List[WeatherImpactResponse])
@create_cache_decorator(ttl_seconds=300)
async def get_weather_impact_analytics(
    entity: str = Query(..., description="Hub or region identifier"),
    window: Optional[str] = Query(None, description="Window label, e.g. 120d"),
    limit: int = Query(5, ge=1, le=50, description="Number of rows to return"),
):
    """Get HDD/CDD weather impact coefficients."""
    try:
        await check_entitlement("gas_data_access")
        clickhouse = await get_clickhouse_client()

        window_clause = ""
        params = {"entity": entity.upper(), "limit": limit}
        if window:
            window_clause = "AND window = %(window)s"
            params["window"] = window

        query = f"""
        SELECT
            date,
            entity_id,
            coef_type,
            coefficient,
            r2,
            window,
            diagnostics
        FROM ch.weather_impact
        WHERE entity_id = %(entity)s
          {window_clause}
        ORDER BY date DESC
        LIMIT %(limit)s
        """

        result = await clickhouse.fetch(query, parameters=params)

        if not result:
            raise HTTPException(status_code=404, detail="No weather impact analytics found")

        responses: List[WeatherImpactResponse] = []
        for row in result:
            responses.append(
                WeatherImpactResponse(
                    as_of_date=row["date"],
                    entity_id=row["entity_id"],
                    coef_type=row["coef_type"],
                    coefficient=row["coefficient"],
                    r2=row.get("r2"),
                    window=row["window"],
                    diagnostics=json.loads(row.get("diagnostics", "{}")),
                )
            )
        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching weather impact analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/analytics/coal-to-gas", response_model=CoalToGasSwitchResponse)
@create_cache_decorator(ttl_seconds=300)
async def get_coal_to_gas_switching(
    region: str = Query(..., description="Region identifier"),
    as_of: date = Query(..., description="As-of date"),
):
    """Get coal-to-gas switching economics."""
    try:
        await check_entitlement("coal_data_access")
        clickhouse = await get_clickhouse_client()

        query = """
        SELECT
            as_of_date,
            region,
            coal_cost_mwh,
            gas_cost_mwh,
            co2_price,
            breakeven_gas_price,
            switch_share,
            diagnostics
        FROM ch.coal_gas_switching
        WHERE region = %(region)s
          AND as_of_date = %(as_of)s
        ORDER BY created_at DESC
        LIMIT 1
        """

        result = await clickhouse.fetch(
            query,
            parameters={"region": region.upper(), "as_of": as_of}
        )

        if not result:
            raise HTTPException(status_code=404, detail="No coal-to-gas analytics found")

        row = result[0]
        return CoalToGasSwitchResponse(
            as_of_date=row["as_of_date"],
            region=row["region"],
            coal_cost_mwh=row["coal_cost_mwh"],
            gas_cost_mwh=row["gas_cost_mwh"],
            co2_price=row["co2_price"],
            breakeven_gas_price=row["breakeven_gas_price"],
            switch_share=row["switch_share"],
            diagnostics=json.loads(row.get("diagnostics", "{}")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching coal-to-gas switching analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/analytics/gas/basis-model", response_model=GasBasisModelResponse)
@create_cache_decorator(ttl_seconds=300)
async def get_gas_basis_model(
    hub: str = Query(..., description="Gas hub identifier"),
    as_of: date = Query(..., description="As-of date"),
):
    """Get gas basis model predictions."""
    try:
        await check_entitlement("gas_data_access")
        clickhouse = await get_clickhouse_client()

        query = """
        SELECT
            as_of_date,
            hub,
            predicted_basis,
            actual_basis,
            method,
            diagnostics,
            feature_snapshot
        FROM ch.gas_basis_models
        WHERE hub = %(hub)s
          AND as_of_date = %(as_of)s
        ORDER BY created_at DESC
        LIMIT 1
        """

        result = await clickhouse.fetch(
            query,
            parameters={"hub": hub.upper(), "as_of": as_of}
        )

        if not result:
            raise HTTPException(status_code=404, detail="No gas basis model output found")

        row = result[0]
        return GasBasisModelResponse(
            as_of_date=row["as_of_date"],
            hub=row["hub"],
            predicted_basis=row["predicted_basis"],
            actual_basis=row.get("actual_basis"),
            method=row["method"],
            diagnostics=json.loads(row.get("diagnostics", "{}")),
            feature_snapshot=json.loads(row.get("feature_snapshot", "{}")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching gas basis model analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/refined-products/prices", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=300)  # 5 minute cache
async def get_refined_products_prices(
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    products: List[str] = Query(default=["RBOB", "ULSD", "JET_FUEL"], description="Refined products to include"),
    locations: List[str] = Query(default=["New York Harbor"], description="Locations to include")
):
    """Get refined petroleum products pricing."""
    try:
        # Check entitlements for refined products data access
        await check_entitlement("refined_products_data_access")

        clickhouse = await get_clickhouse_client()

        # Query refined products prices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'refined_products'
          AND instrument_id IN %(products)s
          AND location_code IN %(locations)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'products': tuple(products),
                'locations': tuple(locations),
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No refined products price data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching refined products prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/biofuels/prices", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=300)  # 5 minute cache
async def get_biofuels_prices(
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    biofuels: List[str] = Query(default=["ETHANOL", "BIODIESEL"], description="Biofuels to include")
):
    """Get biofuels pricing data."""
    try:
        # Check entitlements for biofuels data access
        await check_entitlement("biofuels_data_access")

        clickhouse = await get_clickhouse_client()

        # Query biofuels prices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'biofuels'
          AND instrument_id IN %(biofuels)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'biofuels': tuple(biofuels),
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No biofuels price data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching biofuels prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/emissions/allowances", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=60)  # 1 minute cache for real-time data
async def get_emissions_allowances(
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    markets: List[str] = Query(default=["EUA", "CCA", "RGGI"], description="Carbon markets to include")
):
    """Get carbon allowance pricing."""
    try:
        # Check entitlements for emissions data access
        await check_entitlement("emissions_data_access")

        clickhouse = await get_clickhouse_client()

        # Query carbon allowance prices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'emissions'
          AND instrument_id IN %(markets)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'markets': tuple(markets),
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No emissions allowance data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching emissions allowances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/renewables/certificates", response_model=List[CommodityPriceResponse])
@create_cache_decorator(ttl_seconds=3600)  # 1 hour cache
async def get_renewable_certificates(
    start_date: date = Query(..., description="Start date for price data"),
    end_date: date = Query(..., description="End date for price data"),
    certificates: List[str] = Query(default=["CA_REC", "TX_REC", "SREC"], description="Certificates to include")
):
    """Get renewable energy certificate pricing."""
    try:
        # Check entitlements for renewables data access
        await check_entitlement("renewables_data_access")

        clickhouse = await get_clickhouse_client()

        # Query renewable certificate prices
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp,
            source,
            location_code as location
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'renewables'
          AND instrument_id IN %(certificates)s
          AND event_time >= %(start_date)s
          AND event_time <= %(end_date)s
        ORDER BY event_time DESC
        LIMIT 1000
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'certificates': tuple(certificates),
                'start_date': start_date,
                'end_date': end_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No renewable certificate data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(CommodityPriceResponse(
                commodity_code=row['commodity_code'],
                price=row['price'],
                timestamp=row['timestamp'],
                source=row['source'],
                location=row['location']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching renewable certificates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/benchmarks/comparison", response_model=BenchmarkComparisonResponse)
@create_cache_decorator(ttl_seconds=300)  # 5 minute cache
async def compare_benchmarks(
    benchmark: str = Query(..., description="Primary benchmark commodity"),
    comparison_commodities: List[str] = Query(..., description="Commodities to compare"),
    as_of_date: date = Query(..., description="Comparison date")
):
    """Compare benchmark prices across commodities."""
    try:
        # Check entitlements for benchmark data access
        await check_entitlement("benchmark_data_access")

        clickhouse = await get_clickhouse_client()

        # Query latest prices for comparison
        query = """
        SELECT
            instrument_id as commodity_code,
            value as price,
            event_time as timestamp
        FROM market_intelligence.market_price_ticks
        WHERE instrument_id IN %(all_commodities)s
          AND event_time >= %(as_of_date)s
          AND event_time < %(as_of_date)s + INTERVAL 1 DAY
        ORDER BY event_time DESC
        LIMIT 1000
        """

        all_commodities = [benchmark] + comparison_commodities

        result = await clickhouse.fetch(
            query,
            parameters={
                'all_commodities': tuple(all_commodities),
                'as_of_date': as_of_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No benchmark comparison data found")

        # Group by commodity and get latest price
        latest_prices = {}
        for row in result:
            commodity = row['commodity_code']
            if commodity not in latest_prices:
                latest_prices[commodity] = row['price']

        # Calculate differentials relative to benchmark
        benchmark_price = latest_prices.get(benchmark)
        if benchmark_price is None:
            raise HTTPException(status_code=404, detail=f"Benchmark {benchmark} not found")

        differentials = {}
        for commodity, price in latest_prices.items():
            if commodity != benchmark:
                differential = price - benchmark_price
                differentials[commodity] = differential

        return BenchmarkComparisonResponse(
            benchmark=benchmark,
            comparison_prices=latest_prices,
            price_differentials=differentials,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error comparing benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/futures/curves", response_model=List[FuturesCurveResponse])
@create_cache_decorator(ttl_seconds=1800)  # 30 minute cache
async def get_futures_curves(
    commodities: List[str] = Query(..., description="Commodities to include"),
    as_of_date: date = Query(..., description="As-of date for curves")
):
    """Get futures curves for multiple commodities."""
    try:
        # Check entitlements for futures data access
        await check_entitlement("futures_data_access")

        clickhouse = await get_clickhouse_client()

        # Query futures curves
        query = """
        SELECT
            commodity_code,
            as_of_date,
            contract_month,
            settlement_price,
            open_interest,
            volume,
            exchange
        FROM market_intelligence.futures_curves
        WHERE commodity_code IN %(commodities)s
          AND as_of_date = %(as_of_date)s
        ORDER BY commodity_code, contract_month
        """

        result = await clickhouse.fetch(
            query,
            parameters={
                'commodities': tuple(commodities),
                'as_of_date': as_of_date
            }
        )

        if not result:
            raise HTTPException(status_code=404, detail="No futures curve data found")

        # Convert to response format
        response_data = []
        for row in result:
            response_data.append(FuturesCurveResponse(
                commodity_code=row['commodity_code'],
                as_of_date=row['as_of_date'],
                contract_month=row['contract_month'],
                settlement_price=row['settlement_price'],
                open_interest=row['open_interest'],
                volume=row['volume'],
                exchange=row['exchange']
            ))

        return response_data

    except Exception as e:
        logger.error(f"Error fetching futures curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@commodity_router.get("/specifications/{commodity_code}")
async def get_commodity_specifications(
    commodity_code: str,
    current_user: Dict[str, Any] = Depends(has_permission("read_commodity_specs"))
):
    """Get detailed specifications for a commodity."""
    try:
        clickhouse = await get_clickhouse_client()

        # Query commodity specifications
        query = """
        SELECT
            commodity_code,
            commodity_type,
            contract_unit,
            quality_spec,
            delivery_location,
            exchange
        FROM market_intelligence.commodity_specifications
        WHERE commodity_code = %(commodity_code)s
        """

        result = await clickhouse.fetch_one(
            query,
            parameters={'commodity_code': commodity_code}
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"Commodity specifications not found for {commodity_code}")

        return result

    except Exception as e:
        logger.error(f"Error fetching commodity specifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))
