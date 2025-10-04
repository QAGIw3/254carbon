"""
Automated report generation and delivery service for MISO customers.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import json

from fastapi import BackgroundTasks
from jinja2 import Template

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import db
from cache import cache_response, CacheStrategy

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate automated reports for MISO customers."""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates" / "reports"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Email templates
        self.email_templates = {
            "daily_trading": self._load_template("daily_trading_email.html"),
            "weekly_summary": self._load_template("weekly_summary_email.html"),
        }

    def _load_template(self, template_name: str) -> Template:
        """Load email template."""
        template_path = self.templates_dir / template_name
        if template_path.exists():
            return Template(template_path.read_text())
        else:
            # Fallback to basic template
            return Template(f"<h1>{{{{ title }}}}</h1><p>{{{{ content }}}}</p>")

    async def generate_daily_trading_report(self, report_date: str = None) -> Dict:
        """Generate daily trading report for MISO."""
        if not report_date:
            report_date = (datetime.utcnow() - timedelta(days=1)).date().isoformat()

        try:
            # Get trading data from ClickHouse
            clickhouse = await db.get_clickhouse_client()

            query = """
                SELECT
                    market,
                    instrument_id,
                    SUM(volume) as total_volume,
                    AVG(value) as avg_price,
                    MAX(value) as peak_price,
                    MIN(value) as lowest_price,
                    STDDEV(value) as price_volatility,
                    COUNT(*) as tick_count
                FROM ch.market_price_ticks
                WHERE market = 'MISO'
                    AND event_time >= toDateTime('{report_date}')
                    AND event_time < toDateTime('{report_date}') + INTERVAL 1 DAY
                GROUP BY market, instrument_id
                ORDER BY total_volume DESC
                LIMIT 20
            """

            results = await clickhouse.fetch_all(query)

            # Calculate summary metrics
            total_volume = sum(row['total_volume'] for row in results)
            avg_price = sum(row['avg_price'] * row['total_volume'] for row in results) / total_volume if total_volume > 0 else 0
            peak_price = max((row['peak_price'] for row in results), default=0)
            lowest_price = min((row['lowest_price'] for row in results), default=0)

            report_data = {
                "report_date": report_date,
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_volume_mw": total_volume,
                    "avg_price": avg_price,
                    "peak_price": peak_price,
                    "lowest_price": lowest_price,
                    "price_volatility": sum(row['price_volatility'] for row in results) / len(results) if results else 0,
                    "total_ticks": sum(row['tick_count'] for row in results)
                },
                "top_instruments": [
                    {
                        "instrument_id": row['instrument_id'],
                        "volume": row['total_volume'],
                        "avg_price": row['avg_price'],
                        "peak_price": row['peak_price'],
                        "lowest_price": row['lowest_price'],
                        "volatility": row['price_volatility']
                    }
                    for row in results[:10]  # Top 10
                ],
                "trading_summary": {
                    "total_trades": len(results),
                    "high_volume_threshold": 1000,  # MW
                    "high_volume_trades": len([r for r in results if r['total_volume'] > 1000])
                }
            }

            return report_data

        except Exception as e:
            logger.error(f"Error generating daily trading report: {e}")
            return {"error": str(e)}

    async def generate_weekly_summary_report(self) -> Dict:
        """Generate weekly summary report for MISO."""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=7)

        try:
            # Get weekly data
            clickhouse = await db.get_clickhouse_client()

            query = """
                SELECT
                    toDate(event_time) as date,
                    SUM(volume) as daily_volume,
                    AVG(value) as daily_avg_price,
                    MAX(value) as daily_peak,
                    MIN(value) as daily_low
                FROM ch.market_price_ticks
                WHERE market = 'MISO'
                    AND event_time >= toDateTime('{start_date}')
                    AND event_time < toDateTime('{end_date}') + INTERVAL 1 DAY
                GROUP BY toDate(event_time)
                ORDER BY date
            """

            results = await clickhouse.fetch_all(query)

            # Calculate weekly metrics
            total_volume = sum(row['daily_volume'] for row in results)
            avg_daily_price = sum(row['daily_avg_price'] for row in results) / len(results) if results else 0
            weekly_peak = max((row['daily_peak'] for row in results), default=0)
            weekly_low = min((row['daily_low'] for row in results), default=0)

            # Calculate price trends
            if len(results) >= 2:
                first_price = results[0]['daily_avg_price']
                last_price = results[-1]['daily_avg_price']
                price_change = ((last_price - first_price) / first_price) * 100 if first_price > 0 else 0
            else:
                price_change = 0

            report_data = {
                "report_period": f"{start_date.isoformat()} to {end_date.isoformat()}",
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_volume_mw": total_volume,
                    "avg_daily_price": avg_daily_price,
                    "weekly_peak_price": weekly_peak,
                    "weekly_low_price": weekly_low,
                    "price_change_percent": price_change,
                    "trading_days": len(results)
                },
                "daily_breakdown": [
                    {
                        "date": row['date'].isoformat(),
                        "volume": row['daily_volume'],
                        "avg_price": row['daily_avg_price'],
                        "peak_price": row['daily_peak'],
                        "low_price": row['daily_low']
                    }
                    for row in results
                ],
                "insights": {
                    "price_trend": "up" if price_change > 0 else "down" if price_change < 0 else "stable",
                    "volatility_level": "high" if price_change > 10 else "moderate" if price_change > 5 else "low",
                    "trading_activity": "high" if total_volume > 10000 else "moderate" if total_volume > 5000 else "low"
                }
            }

            return report_data

        except Exception as e:
            logger.error(f"Error generating weekly summary report: {e}")
            return {"error": str(e)}

    async def send_email_report(self, report_data: Dict, recipient: str, report_type: str = "daily_trading"):
        """Send report via email (placeholder for email service integration)."""
        try:
            template = self.email_templates.get(report_type, self.email_templates["daily_trading"])

            # Render email content
            email_content = template.render(
                title=f"MISO {report_type.replace('_', ' ').title()} Report",
                report_data=report_data,
                recipient=recipient
            )

            # Here you would integrate with your email service (SendGrid, SES, etc.)
            # For now, just log the email content
            logger.info(f"Email report for {recipient}: {report_type}")
            logger.info(f"Email content length: {len(email_content)} characters")

            # Store email record in database
            pool = await db.get_postgres_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO miso_email_reports
                    (recipient, report_type, report_data, sent_at, status)
                    VALUES ($1, $2, $3, NOW(), 'sent')
                """, recipient, report_type, json.dumps(report_data))

            return {"status": "sent", "recipient": recipient}

        except Exception as e:
            logger.error(f"Error sending email report: {e}")
            return {"status": "failed", "error": str(e)}

    async def schedule_reports(self):
        """Schedule automated report generation."""
        try:
            # Get customers who have opted in for automated reports
            pool = await db.get_postgres_pool()
            async with pool.acquire() as conn:
                customers = await conn.fetch("""
                    SELECT email, report_preferences
                    FROM miso_customers
                    WHERE automated_reports = true
                """)

                for customer in customers:
                    preferences = json.loads(customer['report_preferences'] or '{}')

                    # Generate daily report if enabled
                    if preferences.get('daily_report', False):
                        report_data = await self.generate_daily_trading_report()
                        if 'error' not in report_data:
                            await self.send_email_report(
                                report_data,
                                customer['email'],
                                'daily_trading'
                            )

                    # Generate weekly report if it's Monday
                    if (datetime.utcnow().weekday() == 0 and  # Monday
                        preferences.get('weekly_report', False)):
                        report_data = await self.generate_weekly_summary_report()
                        if 'error' not in report_data:
                            await self.send_email_report(
                                report_data,
                                customer['email'],
                                'weekly_summary'
                            )

        except Exception as e:
            logger.error(f"Error scheduling reports: {e}")


# Global report generator instance
report_generator = ReportGenerator()


# Background task for automated reports
async def run_scheduled_reports():
    """Run scheduled report generation (to be called by scheduler)."""
    logger.info("Running scheduled report generation...")
    await report_generator.schedule_reports()
    logger.info("Scheduled reports completed")


# API endpoints for report management

@app.post("/api/v1/miso/reports/daily")
async def generate_miso_daily_report(
    report_date: Optional[str] = None,
    email_recipients: List[str] = [],
    user=Depends(verify_token),
):
    """Generate and optionally email daily MISO trading report."""
    track_request("generate_miso_daily_report")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        report_data = await report_generator.generate_daily_trading_report(report_date)

        if 'error' in report_data:
            raise HTTPException(status_code=500, detail=report_data['error'])

        # Send emails if recipients provided
        email_results = []
        if email_recipients:
            for recipient in email_recipients:
                email_result = await report_generator.send_email_report(
                    report_data, recipient, 'daily_trading'
                )
                email_results.append(email_result)

        return {
            "status": "generated",
            "report_data": report_data,
            "email_results": email_results,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating MISO daily report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/miso/reports/weekly")
async def generate_miso_weekly_report(
    email_recipients: List[str] = [],
    user=Depends(verify_token),
):
    """Generate and optionally email weekly MISO summary report."""
    track_request("generate_miso_weekly_report")

    # Check MISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        report_data = await report_generator.generate_weekly_summary_report()

        if 'error' in report_data:
            raise HTTPException(status_code=500, detail=report_data['error'])

        # Send emails if recipients provided
        email_results = []
        if email_recipients:
            for recipient in email_recipients:
                email_result = await report_generator.send_email_report(
                    report_data, recipient, 'weekly_summary'
                )
                email_results.append(email_result)

        return {
            "status": "generated",
            "report_data": report_data,
            "email_results": email_results,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating MISO weekly report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Report email templates (basic HTML templates)

@app.get("/api/v1/miso/reports/templates")
async def get_report_templates(user=Depends(verify_token)):
    """Get available report templates."""
    track_request("get_report_templates")

    return {
        "templates": list(report_generator.email_templates.keys()),
        "description": "Available email templates for MISO reports"
    }
