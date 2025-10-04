"""
ClickHouse query builder and helpers for parameterized, safe, and optimized queries
used by the report-service.

This module centralizes query construction to ensure:
- Parameters are passed separately from SQL to avoid injection
- Consistent WHERE/ORDER/GROUP BY clauses
- Reusable query templates for common report needs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Tuple


@dataclass
class BuiltQuery:
    sql: str
    params: Dict[str, Any]


class ClickHouseQueryBuilder:
    """Minimal fluent builder tailored for our reporting queries.

    Provides a small fluent interface to compose parameterized ClickHouse
    queries with SELECT/FROM/PREWHERE/WHERE/GROUP/ORDER/LIMIT blocks.
    """

    def __init__(self) -> None:
        self._select: List[str] = []
        self._from: str = ""
        self._where: List[str] = []
        self._prewhere: List[str] = []
        self._group_by: List[str] = []
        self._order_by: List[str] = []
        self._limit: int | None = None
        self._params: Dict[str, Any] = {}

    def select(self, *columns: str) -> ClickHouseQueryBuilder:
        """Add columns to the SELECT list.

        Args:
            *columns: Column expressions to select.

        Returns:
            Self for fluent chaining.
        """
        self._select.extend(columns)
        return self

    def from_table(self, table: str) -> ClickHouseQueryBuilder:
        """Set the FROM table.

        Args:
            table: Table name (may include database prefix).

        Returns:
            Self for fluent chaining.
        """
        self._from = table
        return self

    def where(self, condition: str, **params: Any) -> ClickHouseQueryBuilder:
        """Append a WHERE predicate with bound parameters.

        Args:
            condition: SQL predicate using %(name)s placeholders.
            **params: Parameter values to bind.

        Returns:
            Self for fluent chaining.
        """
        self._where.append(condition)
        self._params.update(params)
        return self

    def prewhere(self, condition: str, **params: Any) -> ClickHouseQueryBuilder:
        """Append a PREWHERE predicate with bound parameters.

        Args:
            condition: SQL predicate using %(name)s placeholders.
            **params: Parameter values to bind.

        Returns:
            Self for fluent chaining.
        """
        self._prewhere.append(condition)
        self._params.update(params)
        return self

    def group_by(self, *columns: str) -> ClickHouseQueryBuilder:
        """Append GROUP BY columns.

        Args:
            *columns: Column names/expressions to group by.

        Returns:
            Self for fluent chaining.
        """
        self._group_by.extend(columns)
        return self

    def order_by(self, *columns: str) -> ClickHouseQueryBuilder:
        """Append ORDER BY columns.

        Args:
            *columns: Column names/expressions to order by.

        Returns:
            Self for fluent chaining.
        """
        self._order_by.extend(columns)
        return self

    def limit(self, n: int) -> ClickHouseQueryBuilder:
        """Set LIMIT value.

        Args:
            n: Maximum number of rows to return.

        Returns:
            Self for fluent chaining.
        """
        self._limit = n
        return self

    def add_params(self, **params: Any) -> ClickHouseQueryBuilder:
        """Merge parameters into the current binding map.

        Returns:
            Self for fluent chaining.
        """
        self._params.update(params)
        return self

    def build(self) -> BuiltQuery:
        """Build the SQL string and parameter map.

        Returns:
            BuiltQuery containing the SQL and params dict.

        Raises:
            ValueError: If the FROM table is not specified.
        """
        if not self._from:
            raise ValueError("FROM table must be specified")

        select_sql = ",\n            ".join(self._select) if self._select else "*"

        sql_parts: List[str] = [
            f"SELECT\n            {select_sql}",
            f"FROM {self._from}",
        ]

        if self._prewhere:
            sql_parts.append("PREWHERE " + " AND ".join(self._prewhere))

        if self._where:
            sql_parts.append("WHERE " + " AND ".join(self._where))

        if self._group_by:
            sql_parts.append("GROUP BY " + ", ".join(self._group_by))

        if self._order_by:
            sql_parts.append("ORDER BY " + ", ".join(self._order_by))

        if self._limit is not None:
            sql_parts.append(f"LIMIT {self._limit}")

        sql = "\n".join(sql_parts)
        return BuiltQuery(sql=sql, params=self._params)


def build_price_aggregation_query(
    market: str,
    start_date: date,
    end_date: date,
    table: str = "market_intelligence.market_price_daily_agg",
) -> BuiltQuery:
    """Daily aggregation of prices for a market between dates.

    Args:
        market: Market identifier (e.g., 'MISO').
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        table: Source table name.

    Returns:
        BuiltQuery with SQL and params suitable for ClickHouse driver execute.
    """

    builder = (
        ClickHouseQueryBuilder()
        .select(
            "date",
            "market",
            "instrument_id",
            "avg_price",
            "min_price",
            "max_price",
            "first_price as open_price",
            "last_price as close_price",
            "tick_count as sample_count",
        )
        .from_table(table)
        .prewhere("market = %(market)s", market=market)
        .where("date >= %(start_date)s", start_date=start_date)
        .where("date <= %(end_date)s", end_date=end_date)
        .order_by("date", "instrument_id")
    )

    return builder.build()


def build_forward_curve_query(
    market: str,
    as_of_date: date,
    table: str = "market_intelligence.forward_curve_points",
) -> BuiltQuery:
    """Select forward curve points for a market and as-of date.

    Args:
        market: Market identifier (e.g., 'MISO').
        as_of_date: Valuation date for forward curve.
        table: Source table name.

    Returns:
        BuiltQuery with SQL and params.
    """
    builder = (
        ClickHouseQueryBuilder()
        .select(
            "instrument_id",
            "delivery_start as delivery_period",
            "price",
            "as_of_date",
        )
        .from_table(table)
        .prewhere("market = %(market)s", market=market)
        .where("as_of_date = %(as_of_date)s", as_of_date=as_of_date)
        .order_by("instrument_id", "delivery_period")
    )
    return builder.build()

