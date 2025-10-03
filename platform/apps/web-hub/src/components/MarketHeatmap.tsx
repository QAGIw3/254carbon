import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
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
  onDrillDown?: (region: string, market: string) => void;
}

const MarketHeatmapComponent = ({
  marketData = [],
  height = 400,
  width = 600,
  showTooltips = true,
  onDrillDown
}: MarketHeatmapProps) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedCell, setSelectedCell] = useState<{region: string, market: string} | null>(null);
  const [viewMode, setViewMode] = useState<'price' | 'change'>('price');
  const [isExporting, setIsExporting] = useState(false);

  const handleDrillDown = useCallback((region: string, market: string) => {
    setSelectedCell({region, market});
    onDrillDown?.(region, market);
  }, [onDrillDown]);

  // Memoize data processing for performance
  const processedData = useMemo(() => {
    return marketData.length > 0 ? marketData : generateSampleMarketData();
  }, [marketData]);

  useEffect(() => {
    if (!svgRef.current) return;

    drawHeatmap(svgRef.current, processedData, width, height, viewMode, handleDrillDown);
  }, [processedData, width, height, viewMode, handleDrillDown]);

  const drawHeatmap = (
    svg: SVGSVGElement,
    data: MarketData[],
    width: number,
    height: number,
    viewMode: 'price' | 'change',
    onDrillDown: (region: string, market: string) => void
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

    // Create color scales
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

        // Choose color scale based on view mode
        const colorScale = viewMode === 'price' ? priceScale : changeScale;
        const value = viewMode === 'price' ? cellData.price : cellData.change;

        // Draw cell with enhanced styling and drill-down
        const cell = g.append("rect")
          .attr("x", x)
          .attr("y", y)
          .attr("width", cellWidth - 2)
          .attr("height", cellHeight - 2)
          .attr("fill", colorScale(value))
          .attr("stroke", selectedCell?.region === region && selectedCell?.market === market ? "#333" : "#fff")
          .attr("stroke-width", selectedCell?.region === region && selectedCell?.market === market ? 3 : 1)
          .attr("rx", 4)
          .attr("cursor", "pointer")
          .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))")
          .on("click", () => onDrillDown(region, market))
          .on("mouseover", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("stroke-width", 3)
              .attr("stroke", "#333")
              .style("filter", "drop-shadow(0 4px 8px rgba(0,0,0,0.2))");

            if (showTooltips) {
              showTooltip(d3.event, cellData, viewMode);
            }
          })
          .on("mouseout", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("stroke-width", selectedCell?.region === region && selectedCell?.market === market ? 3 : 1)
              .attr("stroke", selectedCell?.region === region && selectedCell?.market === market ? "#333" : "#fff")
              .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))");

            if (showTooltips) {
              hideTooltip();
            }
          });

        // Add text labels with better contrast based on view mode
        g.append("text")
          .attr("x", x + cellWidth / 2)
          .attr("y", y + cellHeight / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "11px")
          .attr("font-weight", "bold")
          .attr("fill", value > (viewMode === 'price' ? 45 : 0) ? "#fff" : "#333")
          .text(viewMode === 'price' ? `$${cellData.price.toFixed(1)}` : `${cellData.change.toFixed(1)}%`);

        // Add change indicator for price view
        if (viewMode === 'price' && cellData.change !== 0) {
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

  const showTooltip = (event: MouseEvent, data: MarketData, viewMode: 'price' | 'change') => {
    // Create enhanced tooltip with better styling
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.9)")
      .style("color", "white")
      .style("padding", "12px")
      .style("border-radius", "8px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("box-shadow", "0 4px 12px rgba(0,0,0,0.15)")
      .style("border", "1px solid rgba(255,255,255,0.1)");

    const primaryValue = viewMode === 'price' ? `$${data.price.toFixed(2)}` : `${data.change.toFixed(2)}%`;
    const secondaryValue = viewMode === 'price' ? `${data.change > 0 ? '+' : ''}${data.change.toFixed(2)}%` : `$${data.price.toFixed(2)}`;

    tooltip.html(`
      <div style="font-weight: bold; margin-bottom: 4px;">${data.instrument_id}</div>
      <div style="margin-bottom: 2px;">
        <span style="opacity: 0.8;">${viewMode === 'price' ? 'Price:' : 'Change:'}</span>
        <span style="color: ${viewMode === 'price' ? (data.price > 45 ? '#22c55e' : '#ef4444') : (data.change > 0 ? '#22c55e' : '#ef4444')};">${primaryValue}</span>
      </div>
      <div style="margin-bottom: 2px;">
        <span style="opacity: 0.8;">${viewMode === 'price' ? 'Change:' : 'Price:'}</span>
        ${secondaryValue}
      </div>
      <div style="opacity: 0.8; font-size: 11px;">
        ${data.region} • ${data.market}
      </div>
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

  const exportToCSV = () => {
    setIsExporting(true);
    const data = marketData.length > 0 ? marketData : generateSampleMarketData();
    const csvContent = [
      'Region,Market,Price,Change',
      ...data.map(d => `${d.region},${d.market},${d.price},${d.change}`)
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'market-heatmap-data.csv';
    a.click();
    URL.revokeObjectURL(url);
    setIsExporting(false);
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

          {/* Enhanced Controls */}
          <div className="flex gap-2 mt-4 flex-wrap">
            <div className="flex gap-1">
              <button
                className={`px-3 py-1 text-sm border rounded hover:bg-gray-50 ${viewMode === 'price' ? 'bg-blue-50 border-blue-200' : ''}`}
                onClick={() => setViewMode('price')}
              >
                Price View
              </button>
              <button
                className={`px-3 py-1 text-sm border rounded hover:bg-gray-50 ${viewMode === 'change' ? 'bg-green-50 border-green-200' : ''}`}
                onClick={() => setViewMode('change')}
              >
                Change View
              </button>
            </div>
            <button className="px-3 py-1 text-sm border rounded bg-blue-50 border-blue-200">
              Heat Map
            </button>
            <button
              className="px-3 py-1 text-sm border rounded hover:bg-gray-50"
              onClick={exportToCSV}
              disabled={isExporting}
            >
              {isExporting ? 'Exporting...' : 'Export CSV'}
            </button>
          </div>

          {/* Drill-down breadcrumbs */}
          {selectedCell && (
            <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
              <span className="text-sm text-blue-800">
                Selected: {selectedCell.region} • {selectedCell.market}
              </span>
              <button
                className="ml-2 text-xs text-blue-600 hover:text-blue-800"
                onClick={() => setSelectedCell(null)}
              >
                Clear
              </button>
            </div>
          )}

          {/* Info */}
          <div className="mt-4 text-xs text-muted-foreground">
            <p>Click on any cell to drill down. Color intensity represents {viewMode === 'price' ? 'price levels' : 'percentage changes'}.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Memoize component for performance
export default React.memo(MarketHeatmapComponent);
