import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface MarketData {
  instrument_id: string;
  price: number;
  change: number;
  volume?: number;
  region: string;
  market: string;
}

interface MarketHeatmapProps {
  marketData?: MarketData[];
  height?: number;
  width?: number;
  showTooltips?: boolean;
}

export default function MarketHeatmap({
  marketData = [],
  height = 400,
  width = 600,
  showTooltips = true
}: MarketHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    // Generate sample market data if none provided
    const data = marketData.length > 0 ? marketData : generateSampleMarketData();

    drawHeatmap(svgRef.current, data, width, height);
  }, [marketData, width, height]);

  const drawHeatmap = (
    svg: SVGSVGElement,
    data: MarketData[],
    width: number,
    height: number
  ) => {
    // Clear previous content
    d3.select(svg).selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create main group
    const g = d3.select(svg)
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Sample grid data (in real app, this would be geographic regions)
    const regions = ["Northeast", "Midwest", "South", "West", "Northwest"];
    const markets = ["Power", "Gas", "Environmental"];

    // Create color scale based on price levels
    const priceScale = d3.scaleSequential(d3.interpolateRdYlBu)
      .domain([30, 60]); // Price range

    const changeScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([-10, 10]); // Change range

    // Draw grid
    const cellWidth = innerWidth / markets.length;
    const cellHeight = innerHeight / regions.length;

    regions.forEach((region, i) => {
      markets.forEach((market, j) => {
        const x = j * cellWidth;
        const y = i * cellHeight;

        // Find data for this region/market combination
        const cellData = data.find(d =>
          d.region === region && d.market === market
        ) || {
          instrument_id: `${region}_${market}`,
          price: 40 + Math.random() * 20,
          change: (Math.random() - 0.5) * 10,
          region,
          market
        };

        // Draw cell
        g.append("rect")
          .attr("x", x)
          .attr("y", y)
          .attr("width", cellWidth - 2)
          .attr("height", cellHeight - 2)
          .attr("fill", priceScale(cellData.price))
          .attr("stroke", "#fff")
          .attr("stroke-width", 1)
          .attr("rx", 4)
          .attr("cursor", "pointer")
          .on("mouseover", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("stroke-width", 3)
              .attr("stroke", "#333");

            if (showTooltips) {
              showTooltip(d3.event, cellData);
            }
          })
          .on("mouseout", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("stroke-width", 1)
              .attr("stroke", "#fff");

            if (showTooltips) {
              hideTooltip();
            }
          });

        // Add text labels
        g.append("text")
          .attr("x", x + cellWidth / 2)
          .attr("y", y + cellHeight / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "11px")
          .attr("font-weight", "bold")
          .attr("fill", cellData.price > 45 ? "#fff" : "#333")
          .text(`$${cellData.price.toFixed(1)}`);

        // Add change indicator
        if (cellData.change !== 0) {
          g.append("text")
            .attr("x", x + cellWidth - 5)
            .attr("y", y + 12)
            .attr("font-size", "10px")
            .attr("fill", cellData.change > 0 ? "#22c55e" : "#ef4444")
            .attr("font-weight", "bold")
            .text(`${cellData.change > 0 ? '+' : ''}${cellData.change.toFixed(1)}%`);
        }
      });
    });

    // Draw axes
    drawAxes(g, regions, markets, innerWidth, innerHeight);

    // Add legend
    addLegend(g, priceScale, innerWidth, innerHeight);
  };

  const drawAxes = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    regions: string[],
    markets: string[],
    width: number,
    height: number
  ) => {
    // X-axis (Markets)
    g.append("line")
      .attr("x1", 0)
      .attr("y1", height)
      .attr("x2", width)
      .attr("y2", height)
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    // X-axis labels
    markets.forEach((market, i) => {
      const x = (i + 0.5) * (width / markets.length);
      g.append("text")
        .attr("x", x)
        .attr("y", height + 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "12px")
        .text(market);
    });

    // Y-axis (Regions)
    g.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height)
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    // Y-axis labels
    regions.forEach((region, i) => {
      const y = (i + 0.5) * (height / regions.length);
      g.append("text")
        .attr("x", -10)
        .attr("y", y)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "12px")
        .text(region);
    });
  };

  const addLegend = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    colorScale: d3.ScaleSequential<string, never>,
    width: number,
    height: number
  ) => {
    const legendWidth = 200;
    const legendHeight = 20;
    const legendX = width - legendWidth - 20;
    const legendY = height - 40;

    // Legend title
    g.append("text")
      .attr("x", legendX)
      .attr("y", legendY - 10)
      .attr("font-size", "12px")
      .attr("font-weight", "bold")
      .text("Price ($/MWh)");

    // Legend gradient
    const defs = g.append("defs");
    const gradient = defs.append("linearGradient")
      .attr("id", "priceGradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");

    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", d3.interpolateRdYlBu(0));

    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", d3.interpolateRdYlBu(1));

    g.append("rect")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#priceGradient)");

    // Legend labels
    g.append("text")
      .attr("x", legendX)
      .attr("y", legendY + legendHeight + 15)
      .attr("font-size", "11px")
      .text("$30");

    g.append("text")
      .attr("x", legendX + legendWidth)
      .attr("y", legendY + legendHeight + 15)
      .attr("text-anchor", "end")
      .attr("font-size", "11px")
      .text("$60");
  };

  const showTooltip = (event: MouseEvent, data: MarketData) => {
    // Create tooltip (simplified - in real app would use a proper tooltip library)
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", "1000");

    tooltip.html(`
      <strong>${data.instrument_id}</strong><br/>
      Price: $${data.price.toFixed(2)}<br/>
      Change: ${data.change > 0 ? '+' : ''}${data.change.toFixed(2)}%<br/>
      Region: ${data.region}<br/>
      Market: ${data.market}
    `);

    tooltip
      .style("left", (event.pageX + 10) + "px")
      .style("top", (event.pageY - 10) + "px");
  };

  const hideTooltip = () => {
    d3.select(".tooltip").remove();
  };

  const generateSampleMarketData = (): MarketData[] => {
    const regions = ["Northeast", "Midwest", "South", "West", "Northwest"];
    const markets = ["Power", "Gas", "Environmental"];

    const data: MarketData[] = [];

    regions.forEach(region => {
      markets.forEach(market => {
        const basePrice = 35 + Math.random() * 25;
        data.push({
          instrument_id: `${region}_${market}`,
          price: basePrice,
          change: (Math.random() - 0.5) * 10,
          region,
          market
        });
      });
    });

    return data;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Interactive Market Heatmap</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <svg
            ref={svgRef}
            className="w-full border rounded"
            style={{ height: `${height}px` }}
          />

          {/* Controls */}
          <div className="flex gap-2 mt-4">
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Price View
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Change View
            </button>
            <button className="px-3 py-1 text-sm border rounded bg-blue-50 border-blue-200">
              Heat Map
            </button>
          </div>

          {/* Info */}
          <div className="mt-4 text-xs text-muted-foreground">
            <p>Click on any cell to view detailed information. Color intensity represents price levels.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
