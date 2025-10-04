"""
API Marketplace Service

Overview
--------
This FastAPI application exposes a minimal marketplace surface for third‑party
data providers and consumers. It demonstrates core marketplace flows — partner
registration and approval, product listing, sandbox provisioning, usage
tracking, revenue sharing, and basic analytics — using in‑memory state for
illustration purposes. The endpoints and data models are intentionally simple
to act as a reference implementation and are not production‑ready.

Key Concepts
------------
- Partner: an external data provider who lists one or more data products.
- DataProduct: a purchasable/consumable unit of data (e.g., forecasts).
- Usage: metering records for API calls used for billing and analytics.
- RevenueSplit: calculation of partner/platform shares for a period.

Security & Production Notes
---------------------------
- Persistent storage: this module uses in‑memory storage (see `MarketplaceDB`).
  Replace with a persistent database (e.g., PostgreSQL) before production.
- Authentication: partner API keys are pseudo‑random and derived from a hash
  purely for demo; do not use this approach in production. Use a proper secrets
  store and rotation policy, and verify requests via an auth gateway.
- Webhooks: the webhook registration endpoint returns a demo secret; a real
  system should persist and verify webhook signatures on receipt.
- Rate limits: sandbox responses include illustrative limits only; enforce
  limits at an API gateway or service mesh in production.

Developer Experience
--------------------
- Start locally with: `uvicorn main:app --reload --port 8015`.
- Explore the OpenAPI docs at `/docs` or `/redoc` once running.
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
import stripe
import hashlib
import json

# Configure root logger for concise operational visibility. In larger systems,
# prefer structured logging (JSON) + correlation IDs to trace multi‑service calls.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application instance with basic metadata for OpenAPI docs.
app = FastAPI(
    title="API Marketplace",
    description="Third-party integration and monetization platform",
    version="1.0.0",
)


class PartnerStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    SUSPENDED = "suspended"
    REJECTED = "rejected"


class DataProductType(str, Enum):
    MARKET_DATA = "market_data"
    ANALYTICS = "analytics"
    FORECASTS = "forecasts"
    ALERTS = "alerts"


class PricingModel(str, Enum):
    FREE = "free"
    SUBSCRIPTION = "subscription"
    PAY_PER_CALL = "pay_per_call"
    TIERED = "tiered"


class PartnerRegistration(BaseModel):
    """Partner registration request.

    Notes
    -----
    This captures just the data needed to request a partner account. In a
    production environment, you would typically enrich this with company
    identifiers, legal agreements, and verification artifacts.
    """
    company_name: str
    contact_name: str
    email: EmailStr
    website: Optional[str] = None
    description: str
    data_products: List[str]


class Partner(BaseModel):
    """Partner account.

    Attributes
    ----------
    partner_id: Stable identifier assigned at registration/approval time.
    api_key: Per‑partner key for authenticating admin actions. Rotate often.
    revenue_share_pct: Percentage (0‑100) paid to partner from gross revenue.
    status: Lifecycle status for platform governance.
    """
    partner_id: str
    company_name: str
    status: PartnerStatus
    api_key: str
    revenue_share_pct: float
    registered_date: datetime
    approved_date: Optional[datetime] = None


class DataProduct(BaseModel):
    """Data product listing.

    Pricing
    -------
    Only one of `price_per_call` or `monthly_subscription` is typically used,
    depending on the `pricing_model`. `free_tier_calls` is applied per month.
    """
    product_id: str
    partner_id: str
    name: str
    description: str
    category: DataProductType
    pricing_model: PricingModel
    price_per_call: Optional[float] = None
    monthly_subscription: Optional[float] = None
    free_tier_calls: int = 0
    documentation_url: str
    sample_data_url: Optional[str] = None
    status: str  # "active", "pending", "deprecated"


class APIUsage(BaseModel):
    """API usage tracking.

    Each record represents an aggregated count and cost for a time bucket
    (e.g., daily). In production, raw call logs would roll up into this table,
    ideally via a streaming pipeline.
    """
    usage_id: str
    partner_id: str
    product_id: str
    user_id: str
    calls_count: int
    timestamp: datetime
    cost: float


class RevenueSplit(BaseModel):
    """Revenue split calculation.

    The platform takes `(1 - partner_share_pct)` of `total_revenue`.
    """
    period: str  # "2025-10"
    partner_id: str
    total_revenue: float
    partner_share: float
    platform_share: float
    calls_count: int


class SandboxRequest(BaseModel):
    """Sandbox environment request.

    `test_data` can be echoed back by mocked endpoints to facilitate quick
    integration verification without touching real systems.
    """
    partner_id: str
    product_id: str
    test_data: Optional[Dict[str, Any]] = None


# Simple in-memory storage for demo purposes.
#
# WARNING: This is not thread/process safe for real workloads. Swap with a
# transactional store (e.g., PostgreSQL via SQLAlchemy) and add migrations.
class MarketplaceDB:
    def __init__(self):
        self.partners = {}
        self.products = {}
        self.usage = []
        self.next_id = 1

    def save_partner(self, partner: Partner):
        """Persist a partner in the in‑memory index."""
        self.partners[partner.partner_id] = partner.dict()
        return partner

    def get_partner(self, partner_id: str) -> Optional[Partner]:
        """Fetch a partner by identifier or return ``None`` if missing."""
        data = self.partners.get(partner_id)
        if data:
            return Partner(**data)
        return None

    def get_partner_by_api_key(self, api_key: str) -> Optional[Partner]:
        """Lookup a partner by API key.

        Notes
        -----
        A production implementation would index by API key and store hashed
        values only. For simplicity, this performs an O(n) scan.
        """
        for partner_data in self.partners.values():
            if partner_data.get('api_key') == api_key:
                return Partner(**partner_data)
        return None

    def save_product(self, product: DataProduct):
        """Persist a product in the in‑memory catalog."""
        self.products[product.product_id] = product.dict()
        return product

    def get_products(self, category=None, partner_id=None):
        """List active products, optionally filtered by ``category``/``partner_id``."""
        products = []
        for product_data in self.products.values():
            product = DataProduct(**product_data)
            if category and product.category != category:
                continue
            if partner_id and product.partner_id != partner_id:
                continue
            if product.status == "active":
                products.append(product)
        return products

    def record_usage(self, usage: APIUsage):
        """Append a usage record (no deduplication/aggregation performed)."""
        self.usage.append(usage.dict())

    def get_usage(self, partner_id: str, start_date: date, end_date: date):
        """Return usage records for ``partner_id`` within an inclusive date range."""
        usages = []
        for usage_data in self.usage:
            usage_date = datetime.fromisoformat(usage_data['timestamp']).date()
            if (usage_data['partner_id'] == partner_id and
                start_date <= usage_date <= end_date):
                usages.append(APIUsage(**usage_data))
        return usages

# Global database instance
db = MarketplaceDB()


@app.get("/health")
async def health():
    """Lightweight liveness probe used by orchestration layers."""
    return {"status": "healthy"}


@app.post("/api/v1/marketplace/partners/register", response_model=Partner)
async def register_partner(registration: PartnerRegistration):
    """Register a new data provider partner.

    Parameters
    ----------
    registration: PartnerRegistration
        Company and contact metadata for the partner request.

    Returns
    -------
    Partner
        The newly created partner record in ``pending`` status.
    """
    logger.info(f"Partner registration: {registration.company_name}")
    
    # Generate partner ID and API key.
    # This is intentionally simple and deterministic for demo purposes.
    partner_id = f"PTR-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    api_key = f"sk_live_{hash(partner_id + registration.email) % 1000000:06d}"
    
    partner = Partner(
        partner_id=partner_id,
        company_name=registration.company_name,
        status=PartnerStatus.PENDING,
        api_key=api_key,
        revenue_share_pct=70.0,  # Default 70% to partner, 30% to platform
        registered_date=datetime.utcnow(),
    )

    # Save to database.
    db.save_partner(partner)

    return partner


@app.get("/api/v1/marketplace/products", response_model=List[DataProduct])
async def list_products(
    category: Optional[DataProductType] = None,
    partner_id: Optional[str] = None,
):
    """List available data products in the marketplace.

    Filters
    -------
    - ``category``: restrict results to a product category.
    - ``partner_id``: restrict results to a specific partner.
    """
    # Fetch products from database and seed demo products if empty.
    products = db.get_products(category=category, partner_id=partner_id)

    # If no products exist, seed some demo products for discoverability on a
    # fresh environment. In production, remove this block and use fixtures.
    if not products:
        demo_products = [
            DataProduct(
                product_id="PRD-WEATHER-001",
                partner_id="PTR-WEATHERCO",
                name="Weather Impact Forecasts",
                description="ML-powered weather impact on power prices",
                category=DataProductType.FORECASTS,
                pricing_model=PricingModel.SUBSCRIPTION,
                monthly_subscription=299.99,
                free_tier_calls=100,
                documentation_url="https://docs.254carbon.ai/partners/weather",
                status="active",
            ),
            DataProduct(
                product_id="PRD-RENEWABLE-002",
                partner_id="PTR-GREENDATA",
                name="Renewable Generation Forecasts",
                description="Real-time wind and solar forecasts by region",
                category=DataProductType.FORECASTS,
                pricing_model=PricingModel.PAY_PER_CALL,
                price_per_call=0.05,
                free_tier_calls=500,
                documentation_url="https://docs.254carbon.ai/partners/renewable",
                status="active",
            ),
            DataProduct(
                product_id="PRD-EMISSIONS-003",
                partner_id="PTR-CARBONTRACK",
                name="Real-time Emissions Data",
                description="CO2 emissions by generator and market",
                category=DataProductType.MARKET_DATA,
                pricing_model=PricingModel.SUBSCRIPTION,
                monthly_subscription=499.99,
                documentation_url="https://docs.254carbon.ai/partners/emissions",
                status="active",
            ),
        ]

        # Save demo products to database.
        for product in demo_products:
            db.save_product(product)
        products = db.get_products(category=category, partner_id=partner_id)

    return products


@app.post("/api/v1/marketplace/products", response_model=DataProduct)
async def create_product(
    product: DataProduct,
    x_partner_key: str = Header(...),
):
    """Create a new data product for an approved partner.

    Security
    --------
    This endpoint authorizes via the partner API key supplied in header
    ``X-Partner-Key`` (FastAPI maps ``x_partner_key``). In production, verify
    keys through an auth service and enforce scopes (e.g., product:write).
    """
    # Validate partner API key.
    partner = db.get_partner_by_api_key(x_partner_key)
    if not partner:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check if partner is approved.
    if partner.status != PartnerStatus.APPROVED:
        raise HTTPException(status_code=403, detail="Partner account not approved")

    logger.info(f"Creating product: {product.name}")

    # Generate product ID and mark as pending review.
    product.product_id = f"PRD-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    product.partner_id = partner.partner_id
    product.status = "pending"  # Requires approval

    # Save to database.
    db.save_product(product)

    return product


@app.post("/api/v1/marketplace/sandbox")
async def create_sandbox(request: SandboxRequest):
    """Provision a sandbox environment for testing.

    Returns a time‑boxed API key and example request payload. All values are
    illustrative; no backend services are provisioned.
    """
    sandbox_key = f"sk_test_{hash(request.partner_id) % 1000000:06d}"
    
    return {
        "sandbox_id": f"SBX-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "partner_id": request.partner_id,
        "product_id": request.product_id,
        "api_key": sandbox_key,
        "endpoint": "https://sandbox.254carbon.ai/api/v1/",
        "rate_limit": "100 calls/hour",
        "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "sample_requests": [
            {
                "method": "GET",
                "endpoint": "/api/v1/products/{product_id}/data",
                "headers": {"X-API-Key": sandbox_key},
            }
        ],
    }


@app.get("/api/v1/marketplace/usage", response_model=List[APIUsage])
async def get_usage_stats(
    partner_id: str,
    start_date: date,
    end_date: date,
):
    """Get API usage statistics for a partner between two dates (inclusive)."""
    # Get usage from database.
    usage = db.get_usage(partner_id, start_date, end_date)

    # If no usage data exists, generate some mock data for demo.
    if not usage:
        for i in range((end_date - start_date).days + 1):
            usage_date = start_date + timedelta(days=i)
            calls_count = 100 + (hash(str(i)) % 50)
            cost = calls_count * 0.05

            usage_record = APIUsage(
                usage_id=f"USG-{i:06d}",
                partner_id=partner_id,
                product_id="PRD-WEATHER-001",
                user_id=f"USR-{i % 10:03d}",
                calls_count=calls_count,
                timestamp=datetime.combine(usage_date, datetime.min.time()),
                cost=cost,
            )
            db.record_usage(usage_record)

        usage = db.get_usage(partner_id, start_date, end_date)

    return usage


@app.get("/api/v1/marketplace/revenue", response_model=List[RevenueSplit])
async def get_revenue_split(
    partner_id: str,
    year: int,
    month: Optional[int] = None,
):
    """Compute revenue sharing amounts for a partner.

    If ``month`` is omitted, the function returns splits for all months in the
    specified year. Uses usage records as the source of truth and falls back to
    deterministic mock data when none is available.
    """
    if month:
        periods = [f"{year}-{month:02d}"]
    else:
        periods = [f"{year}-{m:02d}" for m in range(1, 13)]

    revenue_splits = []

    # Get partner to check revenue share percentage.
    partner = db.get_partner(partner_id)
    if not partner:
        raise HTTPException(status_code=404, detail="Partner not found")

    partner_share_pct = partner.revenue_share_pct / 100.0

    for period in periods:
        # Calculate revenue based on usage data.
        start_date = date(int(period.split('-')[0]), int(period.split('-')[1]), 1)
        if month:
            end_date = date(int(period.split('-')[0]), int(period.split('-')[1]),
                          (date(int(period.split('-')[0]), int(period.split('-')[1]) + 1, 1) - timedelta(days=1)).day)
        else:
            end_date = date(int(period.split('-')[0]), 12, 31)

        usage_records = db.get_usage(partner_id, start_date, end_date)

        # Calculate total calls and revenue.
        total_calls = sum(usage.calls_count for usage in usage_records)
        total_revenue = sum(usage.cost for usage in usage_records)

        if total_revenue == 0:
            # Generate mock data if no real usage exists.
            total_revenue = 5000.0 + (hash(period) % 2000)
            total_calls = int(total_revenue / 0.05)

        partner_share = total_revenue * partner_share_pct
        platform_share = total_revenue * (1 - partner_share_pct)

        revenue_splits.append(RevenueSplit(
            period=period,
            partner_id=partner_id,
            total_revenue=total_revenue,
            partner_share=partner_share,
            platform_share=platform_share,
            calls_count=total_calls,
        ))

    return revenue_splits


@app.post("/api/v1/marketplace/webhooks")
async def register_webhook(
    partner_id: str,
    webhook_url: str,
    events: List[str],
    x_partner_key: str = Header(...),
):
    """Register a webhook for partner events.

    Example events: ``product_purchased``, ``usage_threshold``, ``payment_received``.
    A production implementation should persist this configuration and send a
    verification ping to the endpoint before activation.
    """
    webhook_id = f"WHK-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    return {
        "webhook_id": webhook_id,
        "partner_id": partner_id,
        "webhook_url": webhook_url,
        "events": events,
        "status": "active",
        "secret": f"whsec_{hash(webhook_id) % 1000000:06d}",
    }


@app.get("/api/v1/marketplace/analytics")
async def get_marketplace_analytics():
    """Return illustrative marketplace‑wide analytics for transparency."""
    return {
        "total_partners": 15,
        "active_products": 42,
        "total_api_calls_month": 1250000,
        "revenue_this_month": 125000.0,
        "top_products": [
            {"product_id": "PRD-WEATHER-001", "name": "Weather Impact", "calls": 250000},
            {"product_id": "PRD-RENEWABLE-002", "name": "Renewable Forecasts", "calls": 180000},
            {"product_id": "PRD-EMISSIONS-003", "name": "Emissions Data", "calls": 120000},
        ],
        "categories": {
            "market_data": 12,
            "analytics": 15,
            "forecasts": 10,
            "alerts": 5,
        },
    }


if __name__ == "__main__":
    import uvicorn
    # Local development entrypoint. In production, run via ASGI server process
    # manager (e.g., uvicorn/gunicorn) and configure workers appropriately.
    uvicorn.run(app, host="0.0.0.0", port=8015)
