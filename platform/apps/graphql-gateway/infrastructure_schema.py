"""
Infrastructure GraphQL Schema Extensions
-----------------------------------------

Adds infrastructure data types and queries to the GraphQL gateway.
"""

from datetime import datetime, date
from typing import List, Optional, Tuple
import json

import strawberry
from strawberry.types import Info
import asyncpg
from clickhouse_driver import Client


# Infrastructure GraphQL Types

@strawberry.type
class InfrastructureAsset:
    """Base infrastructure asset type."""
    asset_id: str
    asset_name: str
    asset_type: str
    country: str
    region: Optional[str]
    latitude: float
    longitude: float
    status: str
    operator: Optional[str]
    owner: Optional[str]
    commissioned_date: Optional[date]
    decommissioned_date: Optional[date]


@strawberry.type
class PowerPlantDetails(InfrastructureAsset):
    """Power plant with generation details."""
    capacity_mw: float
    primary_fuel: str
    secondary_fuel: Optional[str]
    efficiency_pct: Optional[float]
    capacity_factor: Optional[float]
    annual_generation_gwh: Optional[float]
    emissions_rate_tco2_mwh: Optional[float]


@strawberry.type
class LNGTerminalDetails(InfrastructureAsset):
    """LNG terminal with storage details."""
    storage_capacity_gwh: float
    storage_capacity_mcm: Optional[float]
    regasification_capacity_gwh_d: Optional[float]
    send_out_capacity_gwh_d: Optional[float]
    num_tanks: Optional[int]
    berth_capacity: Optional[int]


@strawberry.type
class TransmissionLineDetails(InfrastructureAsset):
    """Transmission line with capacity details."""
    from_latitude: float
    from_longitude: float
    to_latitude: float
    to_longitude: float
    voltage_kv: float
    capacity_mw: float
    length_km: Optional[float]
    line_type: str
    voltage_class: Optional[str]


@strawberry.type
class InfrastructureMetric:
    """Time series metric for infrastructure."""
    ts: datetime
    asset_id: str
    metric: str
    value: float
    unit: str
    metadata: Optional[str]


@strawberry.type
class LNGInventory:
    """LNG terminal inventory snapshot."""
    ts: datetime
    terminal_id: str
    terminal_name: str
    country: str
    inventory_gwh: Optional[float]
    inventory_mcm: Optional[float]
    fullness_pct: Optional[float]
    send_out_gwh: Optional[float]
    ship_arrivals: Optional[int]


@strawberry.type
class PowerGeneration:
    """Power plant generation data."""
    ts: datetime
    plant_id: str
    plant_name: str
    fuel_type: str
    capacity_mw: float
    generation_mwh: Optional[float]
    capacity_factor: Optional[float]
    availability_pct: Optional[float]
    emissions_tco2: Optional[float]


@strawberry.type
class TransmissionFlow:
    """Transmission line flow data."""
    ts: datetime
    line_id: str
    from_zone: str
    to_zone: str
    flow_mw: float
    capacity_mw: float
    utilization_pct: Optional[float]
    congestion_hours: Optional[float]


@strawberry.type
class RenewableResource:
    """Renewable resource assessment."""
    location_id: str
    latitude: float
    longitude: float
    resource_type: str
    annual_average: float
    unit: str
    data_year: int
    resolution_km: float
    monthly_averages: List[float]


@strawberry.type
class InfrastructureProject:
    """Infrastructure project tracking."""
    project_id: str
    project_name: str
    project_type: str
    countries: List[str]
    status: str
    capacity_mw: Optional[float]
    voltage_kv: Optional[float]
    length_km: Optional[float]
    progress_pct: Optional[float]
    estimated_cost_musd: Optional[float]
    start_year: Optional[int]
    completion_year: Optional[int]
    developer: Optional[str]


@strawberry.type
class InfrastructureStats:
    """Aggregated infrastructure statistics."""
    date: date
    country: str
    asset_type: str
    total_capacity: float
    available_capacity: float
    num_assets: int
    num_operational: int
    avg_capacity_factor: Optional[float]


# Infrastructure Query Extensions

@strawberry.type
class InfrastructureQuery:
    """Infrastructure-specific queries."""
    
    @strawberry.field
    async def infrastructure_assets(
        self,
        info: Info,
        asset_type: Optional[str] = None,
        country: Optional[str] = None,
        status: Optional[str] = None,
        min_capacity_mw: Optional[float] = None,
        limit: int = 100,
    ) -> List[InfrastructureAsset]:
        """Search infrastructure assets with filters."""
        
        pool = info.context["pg_pool"]
        
        query = """
            SELECT 
                asset_id, asset_name, asset_type, country, region,
                latitude, longitude, status, operator, owner,
                commissioned_date, decommissioned_date
            FROM pg.infrastructure_assets
            WHERE 1=1
        """
        params = []
        
        if asset_type:
            query += f" AND asset_type = ${len(params) + 1}"
            params.append(asset_type)
        
        if country:
            query += f" AND country = ${len(params) + 1}"
            params.append(country)
        
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
        
        query += f" LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            InfrastructureAsset(
                asset_id=r["asset_id"],
                asset_name=r["asset_name"],
                asset_type=r["asset_type"],
                country=r["country"],
                region=r["region"],
                latitude=float(r["latitude"]),
                longitude=float(r["longitude"]),
                status=r["status"],
                operator=r["operator"],
                owner=r["owner"],
                commissioned_date=r["commissioned_date"],
                decommissioned_date=r["decommissioned_date"],
            )
            for r in rows
        ]
    
    @strawberry.field
    async def power_plants(
        self,
        info: Info,
        country: Optional[str] = None,
        fuel_type: Optional[str] = None,
        min_capacity_mw: Optional[float] = None,
        limit: int = 100,
    ) -> List[PowerPlantDetails]:
        """Get power plants with detailed information."""
        
        pool = info.context["pg_pool"]
        
        query = """
            SELECT 
                a.*, p.capacity_mw, p.primary_fuel, p.secondary_fuel,
                p.efficiency_pct, p.capacity_factor, p.annual_generation_gwh,
                p.emissions_rate_tco2_mwh
            FROM pg.infrastructure_assets a
            JOIN pg.power_plants p ON a.asset_id = p.asset_id
            WHERE a.asset_type = 'power_plant'
        """
        params = []
        
        if country:
            query += f" AND a.country = ${len(params) + 1}"
            params.append(country)
        
        if fuel_type:
            query += f" AND p.primary_fuel = ${len(params) + 1}"
            params.append(fuel_type)
        
        if min_capacity_mw:
            query += f" AND p.capacity_mw >= ${len(params) + 1}"
            params.append(min_capacity_mw)
        
        query += f" ORDER BY p.capacity_mw DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            PowerPlantDetails(
                asset_id=r["asset_id"],
                asset_name=r["asset_name"],
                asset_type=r["asset_type"],
                country=r["country"],
                region=r["region"],
                latitude=float(r["latitude"]),
                longitude=float(r["longitude"]),
                status=r["status"],
                operator=r["operator"],
                owner=r["owner"],
                commissioned_date=r["commissioned_date"],
                decommissioned_date=r["decommissioned_date"],
                capacity_mw=float(r["capacity_mw"]),
                primary_fuel=r["primary_fuel"],
                secondary_fuel=r["secondary_fuel"],
                efficiency_pct=float(r["efficiency_pct"]) if r["efficiency_pct"] else None,
                capacity_factor=float(r["capacity_factor"]) if r["capacity_factor"] else None,
                annual_generation_gwh=float(r["annual_generation_gwh"]) if r["annual_generation_gwh"] else None,
                emissions_rate_tco2_mwh=float(r["emissions_rate_tco2_mwh"]) if r["emissions_rate_tco2_mwh"] else None,
            )
            for r in rows
        ]
    
    @strawberry.field
    async def lng_terminals(
        self,
        info: Info,
        country: Optional[str] = None,
        min_capacity_gwh: Optional[float] = None,
        limit: int = 100,
    ) -> List[LNGTerminalDetails]:
        """Get LNG terminals with storage details."""
        
        pool = info.context["pg_pool"]
        
        query = """
            SELECT 
                a.*, l.storage_capacity_gwh, l.storage_capacity_mcm,
                l.regasification_capacity_gwh_d, l.send_out_capacity_gwh_d,
                l.num_tanks, l.berth_capacity
            FROM pg.infrastructure_assets a
            JOIN pg.lng_terminals l ON a.asset_id = l.asset_id
            WHERE a.asset_type = 'lng_terminal'
        """
        params = []
        
        if country:
            query += f" AND a.country = ${len(params) + 1}"
            params.append(country)
        
        if min_capacity_gwh:
            query += f" AND l.storage_capacity_gwh >= ${len(params) + 1}"
            params.append(min_capacity_gwh)
        
        query += f" ORDER BY l.storage_capacity_gwh DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [
            LNGTerminalDetails(
                asset_id=r["asset_id"],
                asset_name=r["asset_name"],
                asset_type=r["asset_type"],
                country=r["country"],
                region=r["region"],
                latitude=float(r["latitude"]),
                longitude=float(r["longitude"]),
                status=r["status"],
                operator=r["operator"],
                owner=r["owner"],
                commissioned_date=r["commissioned_date"],
                decommissioned_date=r["decommissioned_date"],
                storage_capacity_gwh=float(r["storage_capacity_gwh"]),
                storage_capacity_mcm=float(r["storage_capacity_mcm"]) if r["storage_capacity_mcm"] else None,
                regasification_capacity_gwh_d=float(r["regasification_capacity_gwh_d"]) if r["regasification_capacity_gwh_d"] else None,
                send_out_capacity_gwh_d=float(r["send_out_capacity_gwh_d"]) if r["send_out_capacity_gwh_d"] else None,
                num_tanks=r["num_tanks"],
                berth_capacity=r["berth_capacity"],
            )
            for r in rows
        ]
    
    @strawberry.field
    async def lng_inventory(
        self,
        info: Info,
        start_time: datetime,
        end_time: datetime,
        terminal_id: Optional[str] = None,
        country: Optional[str] = None,
        limit: int = 1000,
    ) -> List[LNGInventory]:
        """Get LNG terminal inventory time series."""
        
        ch_client = info.context["ch_client"]
        
        query = """
            SELECT 
                ts, terminal_id, terminal_name, country,
                inventory_gwh, inventory_mcm, fullness_pct,
                send_out_gwh, ship_arrivals
            FROM market_intelligence.lng_terminal_data
            WHERE ts >= %(start_time)s AND ts <= %(end_time)s
        """
        params = {
            "start_time": start_time,
            "end_time": end_time,
        }
        
        if terminal_id:
            query += " AND terminal_id = %(terminal_id)s"
            params["terminal_id"] = terminal_id
        
        if country:
            query += " AND country = %(country)s"
            params["country"] = country
        
        query += " ORDER BY ts DESC LIMIT %(limit)s"
        params["limit"] = limit
        
        result = ch_client.execute(query, params)
        
        return [
            LNGInventory(
                ts=row[0],
                terminal_id=row[1],
                terminal_name=row[2],
                country=row[3],
                inventory_gwh=row[4],
                inventory_mcm=row[5],
                fullness_pct=row[6],
                send_out_gwh=row[7],
                ship_arrivals=row[8],
            )
            for row in result
        ]
    
    @strawberry.field
    async def power_generation(
        self,
        info: Info,
        start_time: datetime,
        end_time: datetime,
        plant_id: Optional[str] = None,
        fuel_type: Optional[str] = None,
        country: Optional[str] = None,
        limit: int = 1000,
    ) -> List[PowerGeneration]:
        """Get power plant generation time series."""
        
        ch_client = info.context["ch_client"]
        
        query = """
            SELECT 
                ts, plant_id, plant_name, fuel_type, capacity_mw,
                generation_mwh, capacity_factor, availability_pct,
                emissions_tco2
            FROM market_intelligence.power_plant_data
            WHERE ts >= %(start_time)s AND ts <= %(end_time)s
        """
        params = {
            "start_time": start_time,
            "end_time": end_time,
        }
        
        if plant_id:
            query += " AND plant_id = %(plant_id)s"
            params["plant_id"] = plant_id
        
        if fuel_type:
            query += " AND fuel_type = %(fuel_type)s"
            params["fuel_type"] = fuel_type
        
        if country:
            query += " AND country = %(country)s"
            params["country"] = country
        
        query += " ORDER BY ts DESC LIMIT %(limit)s"
        params["limit"] = limit
        
        result = ch_client.execute(query, params)
        
        return [
            PowerGeneration(
                ts=row[0],
                plant_id=row[1],
                plant_name=row[2],
                fuel_type=row[3],
                capacity_mw=row[4],
                generation_mwh=row[5],
                capacity_factor=row[6],
                availability_pct=row[7],
                emissions_tco2=row[8],
            )
            for row in result
        ]
    
    @strawberry.field
    async def renewable_resources(
        self,
        info: Info,
        resource_type: str,
        min_latitude: float,
        max_latitude: float,
        min_longitude: float,
        max_longitude: float,
        limit: int = 1000,
    ) -> List[RenewableResource]:
        """Get renewable resource assessments for a region."""
        
        ch_client = info.context["ch_client"]
        
        query = """
            SELECT 
                location_id, latitude, longitude, resource_type,
                annual_average, unit, data_year, resolution_km,
                monthly_avg_jan, monthly_avg_feb, monthly_avg_mar,
                monthly_avg_apr, monthly_avg_may, monthly_avg_jun,
                monthly_avg_jul, monthly_avg_aug, monthly_avg_sep,
                monthly_avg_oct, monthly_avg_nov, monthly_avg_dec
            FROM market_intelligence.renewable_resources
            WHERE resource_type = %(resource_type)s
                AND latitude >= %(min_lat)s AND latitude <= %(max_lat)s
                AND longitude >= %(min_lon)s AND longitude <= %(max_lon)s
            LIMIT %(limit)s
        """
        params = {
            "resource_type": resource_type,
            "min_lat": min_latitude,
            "max_lat": max_latitude,
            "min_lon": min_longitude,
            "max_lon": max_longitude,
            "limit": limit,
        }
        
        result = ch_client.execute(query, params)
        
        return [
            RenewableResource(
                location_id=row[0],
                latitude=row[1],
                longitude=row[2],
                resource_type=row[3],
                annual_average=row[4],
                unit=row[5],
                data_year=row[6],
                resolution_km=row[7],
                monthly_averages=[row[i] for i in range(8, 20)],
            )
            for row in result
        ]
    
    @strawberry.field
    async def infrastructure_projects(
        self,
        info: Info,
        project_type: Optional[str] = None,
        status: Optional[str] = None,
        countries: Optional[List[str]] = None,
        min_capacity_mw: Optional[float] = None,
        limit: int = 100,
    ) -> List[InfrastructureProject]:
        """Get infrastructure projects with tracking details."""
        
        ch_client = info.context["ch_client"]
        
        query = """
            SELECT 
                project_id, project_name, project_type, countries,
                status, capacity_mw, voltage_kv, length_km,
                progress_pct, estimated_cost_musd, start_year,
                completion_year, developer
            FROM market_intelligence.infrastructure_projects
            WHERE 1=1
        """
        params = {}
        
        if project_type:
            query += " AND project_type = %(project_type)s"
            params["project_type"] = project_type
        
        if status:
            query += " AND status = %(status)s"
            params["status"] = status
        
        if countries:
            query += " AND hasAny(countries, %(countries)s)"
            params["countries"] = countries
        
        if min_capacity_mw:
            query += " AND capacity_mw >= %(min_capacity_mw)s"
            params["min_capacity_mw"] = min_capacity_mw
        
        query += " ORDER BY estimated_cost_musd DESC LIMIT %(limit)s"
        params["limit"] = limit
        
        result = ch_client.execute(query, params)
        
        return [
            InfrastructureProject(
                project_id=row[0],
                project_name=row[1],
                project_type=row[2],
                countries=row[3],
                status=row[4],
                capacity_mw=row[5],
                voltage_kv=row[6],
                length_km=row[7],
                progress_pct=row[8],
                estimated_cost_musd=row[9],
                start_year=row[10],
                completion_year=row[11],
                developer=row[12],
            )
            for row in result
        ]
    
    @strawberry.field
    async def infrastructure_stats(
        self,
        info: Info,
        country: str,
        start_date: date,
        end_date: date,
        asset_type: Optional[str] = None,
    ) -> List[InfrastructureStats]:
        """Get aggregated infrastructure statistics."""
        
        ch_client = info.context["ch_client"]
        
        query = """
            SELECT 
                date, country, asset_type,
                sum(total_capacity) as total_capacity,
                sum(available_capacity) as available_capacity,
                sum(num_assets) as num_assets,
                sum(num_operational) as num_operational,
                avg(avg_capacity_factor) as avg_capacity_factor
            FROM market_intelligence.infrastructure_daily_stats
            WHERE country = %(country)s
                AND date >= %(start_date)s 
                AND date <= %(end_date)s
        """
        params = {
            "country": country,
            "start_date": start_date,
            "end_date": end_date,
        }
        
        if asset_type:
            query += " AND asset_type = %(asset_type)s"
            params["asset_type"] = asset_type
        
        query += " GROUP BY date, country, asset_type ORDER BY date DESC"
        
        result = ch_client.execute(query, params)
        
        return [
            InfrastructureStats(
                date=row[0],
                country=row[1],
                asset_type=row[2],
                total_capacity=row[3],
                available_capacity=row[4],
                num_assets=row[5],
                num_operational=row[6],
                avg_capacity_factor=row[7],
            )
            for row in result
        ]
