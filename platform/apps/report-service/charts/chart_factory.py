"""
Chart factory for Plotly figures used in reports.
Supports multiple chart types with consistent theming and export.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import plotly.graph_objects as go
import plotly.express as px


DEFAULT_TEMPLATE = "plotly_white"


class ChartFactory:
    @staticmethod
    def price_trends(traces: Dict[str, Dict[str, Sequence[Any]]]) -> go.Figure:
        fig = go.Figure()
        for name, series in traces.items():
            fig.add_trace(
                go.Scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            )

        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($/MWh)",
            legend_title="Instrument",
            margin=dict(l=40, r=20, t=40, b=40),
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
        fig = go.Figure(
            data=go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale=colorscale)
        )
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
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


