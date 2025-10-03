"""
Report Service
HTML/PDF generation with charts and monthly market briefs.
"""
import asyncio
import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from clickhouse_driver import Client as ClickHouseClient
from fastapi import FastAPI, HTTPException
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
import boto3

# WeasyPrint import will be done conditionally in functions that need it

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Report Service",
    description="Market report generation",
    version="1.0.0",
)


class ReportRequest(BaseModel):
    report_type: str  # monthly_brief, custom
    market: str
    as_of_date: date
    format: str = "pdf"  # html, pdf


class ReportResponse(BaseModel):
    report_id: str
    status: str
    download_url: Optional[str] = None


async def query_clickhouse_data(market: str, as_of_date: date, report_type: str) -> Dict[str, Any]:
    """Query market data from ClickHouse for report generation."""
    try:
        # Connect to ClickHouse
        client = ClickHouseClient(
            host='clickhouse',
            port=9000,
            database='default',
            user='default',
            password=''
        )

        # Calculate date range based on report type
        if report_type == "monthly_brief":
            start_date = as_of_date.replace(day=1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        else:
            # Custom report - last 30 days
            end_date = as_of_date
            start_date = end_date - timedelta(days=30)

        # Query market price data
        query = """
        SELECT
            toDate(timestamp) as date,
            market,
            instrument_id,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            COUNT(*) as sample_count
        FROM market_price_ticks
        WHERE market = %(market)s
          AND toDate(timestamp) >= %(start_date)s
          AND toDate(timestamp) <= %(end_date)s
        GROUP BY date, market, instrument_id
        ORDER BY date, instrument_id
        """

        price_data = client.execute(
            query,
            market,
            start_date,
            end_date
        )

        # Query forward curve data
        curve_query = """
        SELECT
            instrument_id,
            delivery_period,
            price,
            as_of_date
        FROM forward_curve_points
        WHERE market = %(market)s
          AND as_of_date = %(as_of_date)s
        ORDER BY instrument_id, delivery_period
        """

        curve_data = client.execute(
            curve_query,
            market,
            as_of_date
        )

        return {
            'price_data': price_data,
            'curve_data': curve_data,
            'market': market,
            'start_date': start_date,
            'end_date': end_date,
            'as_of_date': as_of_date,
            'report_type': report_type
        }

    except Exception as e:
        logger.error(f"Error querying ClickHouse: {e}")
        raise


async def generate_charts(data: Dict[str, Any], market: str) -> Dict[str, str]:
    """Generate charts for the report."""
    charts = {}

    try:
        # Price trend chart
        if data['price_data']:
            price_df = data['price_data']
            # Create plotly figure
            fig = go.Figure()

            # Group by instrument and create traces
            instruments = set(row[2] for row in price_df)  # instrument_id is at index 2

            for instrument in list(instruments)[:5]:  # Limit to first 5 instruments
                instrument_data = [row for row in price_df if row[2] == instrument]
                dates = [row[0] for row in instrument_data]
                prices = [row[3] for row in instrument_data]  # avg_price is at index 3

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines+markers',
                    name=instrument,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))

            fig.update_layout(
                title=f'{market} Price Trends',
                xaxis_title='Date',
                yaxis_title='Price ($/MWh)',
                template='plotly_white',
                height=400,
                showlegend=True
            )

            # Convert to HTML
            charts['price_trend'] = fig.to_html(full_html=False, include_plotlyjs=False)

        # Forward curve chart
        if data['curve_data']:
            curve_df = data['curve_data']
            if curve_df:
                # Create forward curve chart
                fig_curve = go.Figure()

                # Group by instrument
                instruments = {}
                for row in curve_df:
                    instrument_id = row[0]
                    if instrument_id not in instruments:
                        instruments[instrument_id] = []
                    instruments[instrument_id].append((row[1], row[2]))  # (delivery_period, price)

                for instrument, points in list(instruments.items())[:3]:  # Limit to 3 instruments
                    points.sort(key=lambda x: x[0])  # Sort by delivery period
                    periods = [p[0] for p in points]
                    prices = [p[1] for p in points]

                    fig_curve.add_trace(go.Scatter(
                        x=periods,
                        y=prices,
                        mode='lines+markers',
                        name=instrument,
                        line=dict(width=3),
                        marker=dict(size=6)
                    ))

                fig_curve.update_layout(
                    title=f'{market} Forward Curves',
                    xaxis_title='Delivery Period',
                    yaxis_title='Price ($/MWh)',
                    template='plotly_white',
                    height=400,
                    showlegend=True
                )

                charts['forward_curve'] = fig_curve.to_html(full_html=False, include_plotlyjs=False)

        return charts

    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        raise


async def render_html_template(
    request: ReportRequest,
    data: Dict[str, Any],
    charts: Dict[str, str]
) -> str:
    """Render HTML template with data and charts."""
    try:
        # Setup Jinja2 environment
        env = Environment(loader=FileSystemLoader('templates'))

        # Load template from file
        template = env.get_template('market_report.html')

        # Calculate comprehensive statistics
        stats = await calculate_report_statistics(data, request.market)

        # Render template
        html_content = template.render(
            title=f"{request.market.upper()} {request.report_type.replace('_', ' ').title()}",
            market=request.market.upper(),
            as_of_date=request.as_of_date,
            start_date=data['start_date'],
            end_date=data['end_date'],
            generation_time=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
            avg_price=stats['avg_price'],
            min_price=stats['min_price'],
            max_price=stats['max_price'],
            total_volume=stats['total_volume'],
            volatility=stats['volatility'],
            charts=charts,
            price_data=data['price_data'][:25] if data['price_data'] else [],
            curve_data=data['curve_data'][:20] if data['curve_data'] else []
        )

        return html_content

    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise


async def calculate_report_statistics(data: Dict[str, Any], market: str) -> Dict[str, Any]:
    """Calculate comprehensive report statistics."""
    try:
        if not data['price_data']:
            return {
                'avg_price': 0,
                'min_price': 0,
                'max_price': 0,
                'total_volume': 0,
                'volatility': 0
            }

        # Extract price data
        all_prices = []
        for row in data['price_data']:
            if row[3] is not None:  # avg_price is at index 3
                all_prices.append(float(row[3]))

        if not all_prices:
            return {
                'avg_price': 0,
                'min_price': 0,
                'max_price': 0,
                'total_volume': 0,
                'volatility': 0
            }

        avg_price = sum(all_prices) / len(all_prices)
        min_price = min(all_prices)
        max_price = max(all_prices)
        total_volume = len(data['price_data'])

        # Calculate volatility (standard deviation as percentage of mean)
        if avg_price > 0:
            price_std = np.std(all_prices)
            volatility = (price_std / avg_price) * 100
        else:
            volatility = 0

        return {
            'avg_price': avg_price,
            'min_price': min_price,
            'max_price': max_price,
            'total_volume': total_volume,
            'volatility': volatility
        }

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {
            'avg_price': 0,
            'min_price': 0,
            'max_price': 0,
            'total_volume': 0,
            'volatility': 0
        }


async def generate_pdf(html_content: str) -> bytes:
    """Generate PDF from HTML content."""
    try:
        # Import WeasyPrint conditionally here to avoid module-level import issues
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            weasyprint_available = True
        except ImportError:
            weasyprint_available = False

        if not weasyprint_available:
            logger.warning("WeasyPrint not available - falling back to HTML-only report")
            fallback_html = """
            <html>
            <head><title>Report Generation - PDF Unavailable</title></head>
            <body style="font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto;">
            <h1>📄 Report Generation</h1>
            <div style="background: #f8f9fa; border-left: 4px solid #ffc107; padding: 20px; margin: 20px 0; border-radius: 5px;">
            <h3 style="margin-top: 0;">⚠️ PDF Generation Unavailable</h3>
            <p>PDF generation requires system dependencies that are not currently installed.</p>
            <p><strong>Required packages:</strong></p>
            <ul>
            <li>libpango1.0-dev</li>
            <li>libharfbuzz-dev</li>
            <li>libfribidi-dev</li>
            <li>libcairo2-dev</li>
            </ul>
            <p>HTML reports are fully functional. For PDF support, please install the required system libraries.</p>
            </div>
            <h2>📋 HTML Report Preview</h2>
            <p>The complete HTML report is available for download. This report includes:</p>
            <ul>
            <li>📊 Executive summary with key metrics</li>
            <li>📈 Interactive price trend charts</li>
            <li>🔮 Forward curve analysis</li>
            <li>📋 Detailed price data tables</li>
            <li>💡 Market insights and recommendations</li>
            </ul>
            <p><em>Report generated by 254Carbon Market Intelligence Platform</em></p>
            </body>
            </html>
            """
            return fallback_html.encode('utf-8')

        font_config = FontConfiguration()
        html_doc = HTML(string=html_content)

        # CSS for better PDF formatting
        css = CSS(string='''
            @page {
                size: A4;
                margin: 1in;
                @bottom-right {
                    content: "Page " counter(page) " of " counter(pages);
                }
            }
            body {
                font-size: 12px;
                line-height: 1.4;
            }
            .header {
                page-break-after: avoid;
            }
            .section {
                page-break-inside: avoid;
                margin-bottom: 20px;
            }
        ''')

        pdf_bytes = html_doc.write_pdf(stylesheets=[css], font_config=font_config)
        return pdf_bytes

    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        logger.info("Falling back to HTML-only report")
        # Return a simple fallback message for PDF
        fallback_html = f"""
        <html>
        <head><title>Report Generation Error</title></head>
        <body>
        <h1>Report Generation</h1>
        <p>PDF generation failed due to missing system dependencies.</p>
        <p>HTML report is available. Please install system dependencies for PDF support:</p>
        <ul>
        <li>libpango1.0-dev</li>
        <li>libharfbuzz-dev</li>
        <li>libfribidi-dev</li>
        </ul>
        <p>Error: {str(e)}</p>
        </body>
        </html>
        """
        return fallback_html.encode('utf-8')


async def store_in_minio(report_id: str, content: bytes, extension: str, content_type: str) -> str:
    """Store report content in MinIO."""
    try:
        # Connect to MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9001',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1'
        )

        # Create bucket if it doesn't exist
        bucket_name = 'reports'
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except:
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload file
        filename = f"{report_id}.{extension}"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=content,
            ContentType=content_type,
            ACL='public-read'  # Make publicly accessible for download
        )

        # Return public URL
        return f"http://minio:9001/{bucket_name}/{filename}"

    except Exception as e:
        logger.error(f"Error storing in MinIO: {e}")
        raise


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/reports", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """Generate a market report."""
    report_id = str(uuid.uuid4())

    logger.info(
        f"Generating {request.report_type} report for {request.market}, "
        f"as of {request.as_of_date}"
    )

    try:
        # Query data from ClickHouse
        clickhouse_data = await query_clickhouse_data(
            request.market, request.as_of_date, request.report_type
        )

        # Generate charts
        charts = await generate_charts(clickhouse_data, request.market)

        # Render HTML template
        html_content = await render_html_template(
            request, clickhouse_data, charts
        )

        # Convert to PDF if requested
        if request.format == "pdf":
            pdf_content = await generate_pdf(html_content)
            file_extension = "pdf"
            content_type = "application/pdf"
        else:
            pdf_content = html_content.encode('utf-8')
            file_extension = "html"
            content_type = "text/html"

        # Store in MinIO
        minio_url = await store_in_minio(
            report_id, pdf_content, file_extension, content_type
        )

        return ReportResponse(
            report_id=report_id,
            status="completed",
            download_url=minio_url,
        )

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str):
    """Get report status and download URL."""
    try:
        # Connect to MinIO to check if report exists
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9001',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1'
        )

        bucket_name = 'reports'
        filename = f"{report_id}.pdf"

        # Check if PDF exists
        try:
            s3_client.head_object(Bucket=bucket_name, Key=filename)
            download_url = f"http://minio:9001/{bucket_name}/{filename}"
            status = "completed"
        except:
            # Check if HTML exists
            filename_html = f"{report_id}.html"
            try:
                s3_client.head_object(Bucket=bucket_name, Key=filename_html)
                download_url = f"http://minio:9001/{bucket_name}/{filename_html}"
                status = "completed"
            except:
                download_url = None
                status = "not_found"

        return ReportResponse(
            report_id=report_id,
            status=status,
            download_url=download_url,
        )

    except Exception as e:
        logger.error(f"Error getting report status: {e}")
        return ReportResponse(
            report_id=report_id,
            status="error",
            download_url=None,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

