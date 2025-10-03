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
from clickhouse_driver import Client as ClickHouseClient
from fastapi import FastAPI, HTTPException
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
import boto3
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

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
            parameters={
                'market': market,
                'start_date': start_date,
                'end_date': end_date
            }
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
            parameters={
                'market': market,
                'as_of_date': as_of_date
            }
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

        # For now, create inline template since templates directory might not exist
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
                .section { margin: 30px 0; }
                .chart-container { margin: 20px 0; text-align: center; }
                .summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
                .stat-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; }
                .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .stat-label { color: #7f8c8d; margin-top: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f5f5f5; font-weight: bold; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p><strong>Market:</strong> {{ market }}</p>
                <p><strong>Report Date:</strong> {{ as_of_date }}</p>
                <p><strong>Period:</strong> {{ start_date }} to {{ end_date }}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-value">{{ avg_price|round(2) }}</div>
                        <div class="stat-label">Average Price ($/MWh)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ price_range }}</div>
                        <div class="stat-label">Price Range ($/MWh)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ total_volume }}</div>
                        <div class="stat-label">Sample Count</div>
                    </div>
                </div>
            </div>

            {% if charts.price_trend %}
            <div class="section">
                <h2>Price Trends</h2>
                <div class="chart-container">
                    {{ charts.price_trend|safe }}
                </div>
            </div>
            {% endif %}

            {% if charts.forward_curve %}
            <div class="section">
                <h2>Forward Curves</h2>
                <div class="chart-container">
                    {{ charts.forward_curve|safe }}
                </div>
            </div>
            {% endif %}

            {% if price_data %}
            <div class="section">
                <h2>Price Data Summary</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Instrument</th>
                            <th>Avg Price</th>
                            <th>Min Price</th>
                            <th>Max Price</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in price_data[:20] %}
                        <tr>
                            <td>{{ row[0] }}</td>
                            <td>{{ row[2] }}</td>
                            <td>{{ row[3]|round(2) }}</td>
                            <td>{{ row[4]|round(2) }}</td>
                            <td>{{ row[5]|round(2) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% if price_data|length > 20 %}
                <p><em>Showing first 20 records. Total: {{ price_data|length }} records.</em></p>
                {% endif %}
            </div>
            {% endif %}
        </body>
        </html>
        """

        # Create inline template
        template = env.from_string(template_str)

        # Calculate summary statistics
        if data['price_data']:
            all_prices = [row[3] for row in data['price_data'] if row[3] is not None]
            avg_price = sum(all_prices) / len(all_prices) if all_prices else 0

            min_price = min(all_prices) if all_prices else 0
            max_price = max(all_prices) if all_prices else 0
            price_range = f"{min_price:.2f} - {max_price:.2f}"

            total_volume = len(data['price_data'])
        else:
            avg_price = 0
            price_range = "N/A"
            total_volume = 0

        # Render template
        html_content = template.render(
            title=f"{request.market} {request.report_type.replace('_', ' ').title()}",
            market=request.market,
            as_of_date=request.as_of_date,
            start_date=data['start_date'],
            end_date=data['end_date'],
            avg_price=avg_price,
            price_range=price_range,
            total_volume=total_volume,
            charts=charts,
            price_data=data['price_data'][:20] if data['price_data'] else []
        )

        return html_content

    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise


async def generate_pdf(html_content: str) -> bytes:
    """Generate PDF from HTML content."""
    try:
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
        logger.error(f"Error generating PDF: {e}")
        raise


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

