import pytest
from unittest.mock import Mock, patch
from report_service.charts.chart_factory import ChartFactory


def test_price_trends_chart_creation():
    """Test that price trends chart is created correctly."""
    traces = {
        "MISO.HUB.INDIANA": {
            "x": ["2024-01-01", "2024-01-02"],
            "y": [35.5, 36.2]
        },
        "PJM.HUB.WEST": {
            "x": ["2024-01-01", "2024-01-02"],
            "y": [40.1, 39.8]
        }
    }

    fig = ChartFactory.price_trends(traces)

    # Check that figure has traces
    assert len(fig.data) == 2
    assert fig.data[0].name == "MISO.HUB.INDIANA"
    assert fig.data[1].name == "PJM.HUB.WEST"

    # Check layout
    assert fig.layout.height == 400
    assert fig.layout.xaxis.title.text == "Date"
    assert fig.layout.yaxis.title.text == "Price ($/MWh)"


def test_candlestick_chart_creation():
    """Test candlestick chart creation."""
    dates = ["2024-01-01", "2024-01-02"]
    open_prices = [35.0, 36.0]
    high_prices = [37.0, 38.0]
    low_prices = [33.0, 34.0]
    close_prices = [36.0, 37.0]

    fig = ChartFactory.candlestick(dates, open_prices, high_prices, low_prices, close_prices)

    assert len(fig.data) == 1
    assert fig.data[0].type == "candlestick"
    assert fig.layout.height == 400


def test_correlation_matrix_chart():
    """Test correlation matrix chart creation."""
    matrix = [
        [1.0, 0.8, -0.2],
        [0.8, 1.0, 0.1],
        [-0.2, 0.1, 1.0]
    ]
    labels = ["A", "B", "C"]

    fig = ChartFactory.correlation_matrix(matrix, labels)

    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert fig.layout.title.text == "Correlation Matrix"


def test_chart_factory_consistent_theming():
    """Test that all charts use consistent theming."""
    # Test price trends
    traces = {"Test": {"x": [1, 2], "y": [1, 2]}}
    fig1 = ChartFactory.price_trends(traces)
    assert fig1.layout.template == "plotly_white"

    # Test candlestick
    fig2 = ChartFactory.candlestick([1, 2], [1, 1], [2, 2], [0, 0], [1, 1])
    assert fig2.layout.template == "plotly_white"
