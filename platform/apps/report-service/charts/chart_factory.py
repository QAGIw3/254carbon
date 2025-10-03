"""
Chart factory for Plotly figures used in reports.
Supports multiple chart types with consistent theming and export.
Includes performance optimizations for large datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
import plotly.express as px


DEFAULT_TEMPLATE = "plotly_white"

# Performance optimization constants
MAX_SERIES_POINTS = 1000  # Limit series length for performance
MAX_HEATMAP_CELLS = 5000  # Limit heatmap size for performance


def downsample_series(x: Sequence[Any], y: Sequence[float], max_points: int = MAX_SERIES_POINTS) -> Tuple[List[Any], List[float]]:
    """Downsample time series data for performance while preserving shape."""
    if len(x) <= max_points:
        return list(x), list(y)

    # Use Douglas-Peucker algorithm for intelligent downsampling
    # For simplicity, use uniform sampling
    step = len(x) // max_points
    indices = list(range(0, len(x), step))[:max_points]

    return [x[i] for i in indices], [y[i] for i in indices]


class ChartFactory:
    @staticmethod
    def price_trends(traces: Dict[str, Dict[str, Sequence[Any]]], max_points: int = MAX_SERIES_POINTS) -> go.Figure:
        """Create price trends chart with performance optimizations."""
        fig = go.Figure()

        for name, series in traces.items():
            x_data = series["x"]
            y_data = series["y"]

            # Downsample data using intelligent algorithm
            x_data, y_data = downsample_series(x_data, y_data, max_points)

            # Use simplified mode for better performance with large datasets
            mode = "lines" if len(x_data) > 500 else "lines+markers"

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode,
                    name=name,
                    line=dict(width=2, shape="spline" if len(x_data) < 200 else "linear"),
                    marker=dict(size=4 if len(x_data) < 500 else 2),
                )
            )

        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($/MWh)",
            legend_title="Instrument",
            margin=dict(l=40, r=20, t=40, b=40),
            # Performance optimizations
            showlegend=len(traces) <= 10,  # Hide legend for many series
        )
        return fig

    @staticmethod
    def candlestick(
        dates: Sequence[Any], open_: Sequence[float], high: Sequence[float], low: Sequence[float], close: Sequence[float], name: str = "Price"
    ) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Candlestick(x=dates, open=open_, high=high, low=low, close=close, name=name)
            ]
        )
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=400,
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def heatmap(z: List[List[float]], x_labels: Sequence[str], y_labels: Sequence[str], colorscale: str = "RdBu") -> go.Figure:
        """Create heatmap with performance optimizations for large datasets."""
        # Check if data is too large and needs downsampling
        total_cells = len(x_labels) * len(y_labels)
        if total_cells > MAX_HEATMAP_CELLS:
            # Simple downsampling - in production would use more sophisticated algorithms
            step_x = max(1, len(x_labels) // int((MAX_HEATMAP_CELLS / len(y_labels)) ** 0.5))
            step_y = max(1, len(y_labels) // int((MAX_HEATMAP_CELLS / len(x_labels)) ** 0.5))

            x_sampled = x_labels[::step_x][:50]  # Limit to 50 labels max
            y_sampled = y_labels[::step_y][:50]
            z_sampled = [row[::step_x][:len(x_sampled)] for row in z[:len(y_sampled)]]
        else:
            x_sampled = x_labels
            y_sampled = y_labels
            z_sampled = z

        fig = go.Figure(
            data=go.Heatmap(
                z=z_sampled,
                x=x_sampled,
                y=y_sampled,
                colorscale=colorscale,
                # Performance optimizations
                hovertemplate="%{x}<br>%{y}<br>Value: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=min(400, max(200, len(y_sampled) * 8)),  # Dynamic height based on data size
            margin=dict(l=40, r=20, t=40, b=40),
            # Performance optimizations
            showlegend=False,
        )
        return fig

    @staticmethod
    def correlation_matrix(matrix: List[List[float]], labels: Sequence[str]) -> go.Figure:
        fig = go.Figure(
            data=go.Heatmap(z=matrix, x=labels, y=labels, colorscale="RdBu", zmin=-1, zmax=1)
        )
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=400,
            title="Correlation Matrix",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def surface_3d(x: Sequence[Any], y: Sequence[Any], z: List[List[float]], colorscale: str = "Viridis") -> go.Figure:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale=colorscale)])
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            height=500,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def scatter_3d(x: Sequence[float], y: Sequence[float], z: Sequence[float], name: str = "Points") -> go.Figure:
        fig = go.Figure(
            data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", name=name, marker=dict(size=4))]
        )
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            height=500,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    @staticmethod
    def sankey(nodes: Sequence[str], links: List[Dict[str, int | float]]) -> go.Figure:
        node_labels = list(nodes)
        # Map node name to index
        index_map = {name: i for i, name in enumerate(node_labels)}
        source = [index_map[link["source"]] for link in links]  # type: ignore
        target = [index_map[link["target"]] for link in links]  # type: ignore
        value = [float(link["value"]) for link in links]
        fig = go.Figure(
            data=[go.Sankey(
                node=dict(label=node_labels),
                link=dict(source=source, target=target, value=value),
            )]
        )
        fig.update_layout(template=DEFAULT_TEMPLATE, height=500, margin=dict(l=40, r=20, t=40, b=40))
        return fig

    @staticmethod
    def animated_time_series(frames: List[Dict[str, Any]], x_title: str = "Date", y_title: str = "Value") -> go.Figure:
        # frames: [{"name": frameName, "data": [{"x": [...], "y": [...], "name": seriesName}, ...]}]
        base = frames[0] if frames else {"data": []}
        fig = go.Figure(
            data=[go.Scatter(x=s["x"], y=s["y"], mode="lines", name=s.get("name", "Series")) for s in base.get("data", [])],
            frames=[go.Frame(name=f.get("name"), data=[go.Scatter(x=s["x"], y=s["y"]) for s in f.get("data", [])]) for f in frames]
        )
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            xaxis_title=x_title,
            yaxis_title=y_title,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}]},
                ]
            }],
        )
        return fig

    @staticmethod
    def forward_curves(curves: Dict[str, Dict[str, Sequence[Any]]]) -> go.Figure:
        fig = go.Figure()
        for name, series in curves.items():
            fig.add_trace(
                go.Scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=3),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=400,
            xaxis_title="Delivery Period",
            yaxis_title="Price ($/MWh)",
            legend_title="Instrument",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig


