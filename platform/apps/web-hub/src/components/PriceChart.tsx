import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface PriceData {
  timestamp: string;
  price: number;
  volume?: number;
}

interface PriceChartProps {
  data: PriceData[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  color?: string;
}

export default function PriceChart({
  data,
  width = 800,
  height = 400,
  margin = { top: 20, right: 30, bottom: 40, left: 50 },
  color = '#3b82f6'
}: PriceChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const cleanup = drawChart();
    return cleanup;
  }, [data, width, height, margin, color]);

  const drawChart = () => {
    const svgElement = svgRef.current;
    if (!svgElement) return () => {};
    const svgSelection = d3.select(svgElement as SVGSVGElement);
    svgSelection.selectAll('*').remove();

    const svg = svgSelection
      .attr('width', width)
      .attr('height', height);

    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Create scales
    const extent = d3.extent(data, d => new Date(d.timestamp)) as [Date, Date] | [undefined, undefined];
    const validExtent: [Date, Date] = [extent[0] ?? new Date(), extent[1] ?? new Date()];

    const xScale = d3
      .scaleTime()
      .domain(validExtent)
      .range([0, chartWidth]);

    const minPrice = d3.min(data, d => d.price) ?? 0;
    const maxPrice = d3.max(data, d => d.price) ?? 0;
    const yScale = d3
      .scaleLinear()
      .domain([
        minPrice * 0.95,
        maxPrice * 1.05
      ])
      .range([chartHeight, 0]);

    // Create line generator
    const line = d3
      .line<PriceData>()
      .defined(d => !Number.isNaN(d.price))
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.price))
      .curve(d3.curveMonotoneX);

    // Create area generator for confidence bands (if volume data available)
    const hasVolume = data.some(d => d.volume !== undefined);
    const area = hasVolume
      ? d3
          .area<PriceData>()
          .x(d => xScale(new Date(d.timestamp)))
          .y0(d => yScale(d.price * (1 - ((d.volume ?? 0) / 1000))))
          .y1(d => yScale(d.price * (1 + ((d.volume ?? 0) / 1000))))
          .curve(d3.curveMonotoneX)
      : null;

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add grid lines
    g.append('g')
      .attr('class', 'grid x-axis-grid')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(
        d3.axisBottom(xScale)
          .tickSize(-chartHeight)
          .tickFormat(() => '') as any
      )
      .selectAll('line')
      .style('stroke', '#e5e7eb')
      .style('stroke-width', 0.5);

    g.append('g')
      .attr('class', 'grid y-axis-grid')
      .call(
        d3.axisLeft(yScale)
          .tickSize(-chartWidth)
          .tickFormat(() => '') as any
      )
      .selectAll('line')
      .style('stroke', '#e5e7eb')
      .style('stroke-width', 0.5);

    // Add axes
    const xAxisGenerator = d3.axisBottom(xScale).ticks(5);
    const axisX = g.append<SVGGElement>('g')
      .attr('class', 'axis--x')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(xAxisGenerator as any);

    const yAxisGenerator = d3.axisLeft(yScale).ticks(5);
    g.append<SVGGElement>('g')
      .attr('class', 'axis--y')
      .call(yAxisGenerator as any)
      .selectAll('text')
      .style('text-anchor', 'end');

    // Add area for volume-based confidence bands
    if (area) {
      const areaPath = area(data as PriceData[]);
      if (areaPath) {
        g.append('path')
          .attr('class', 'area')
          .attr('d', areaPath)
          .style('fill', color)
          .style('opacity', 0.1);
      }
    }

    // Add price line
    const linePath = line(data as PriceData[]);
    if (linePath) {
      g.append('path')
        .attr('class', 'line')
        .attr('d', linePath)
        .style('fill', 'none')
        .style('stroke', color)
        .style('stroke-width', 2);
    }

    // Add data points
    g.selectAll<SVGCircleElement, PriceData>('.point')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('cx', d => xScale(new Date(d.timestamp)))
      .attr('cy', d => yScale(d.price))
      .attr('r', 3)
      .style('fill', color)
      .style('stroke', '#fff')
      .style('stroke-width', 2);

    // Add hover effects and tooltips
    const tooltip = d3.select<HTMLDivElement, unknown>('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    g.selectAll<SVGCircleElement, PriceData>('.point')
      .on('mouseover', function(event: MouseEvent, d: PriceData) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 6);

        tooltip
          .style('opacity', 1)
          .html(`
            <strong>Price:</strong> $${d.price.toFixed(2)}<br/>
            <strong>Time:</strong> ${new Date(d.timestamp).toLocaleString()}<br/>
            ${d.volume ? `<strong>Volume:</strong> ${d.volume} MW` : ''}
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 3);

        tooltip.style('opacity', 0);
      });

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .translateExtent([[0, 0], [width, height]])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        const newXScale = event.transform.rescaleX(xScale);

        const updatedLine = line.x(d => newXScale(new Date(d.timestamp)));
        g.select('.line').attr('d', updatedLine as any);
        if (area) {
          const updatedArea = area.x(d => newXScale(new Date(d.timestamp)));
          g.select<SVGPathElement>('.area').attr('d', updatedArea as any);
        }

        g.selectAll<SVGCircleElement, PriceData>('.point')
          .attr('cx', d => newXScale(new Date(d.timestamp)))
          .attr('cy', d => yScale(d.price));

        g.select<SVGGElement>('.grid.x-axis-grid')
          .call(
            d3.axisBottom(newXScale)
              .tickSize(-chartHeight)
              .tickFormat(() => '') as any
          );

        axisX.call(d3.axisBottom(newXScale).ticks(5) as any);
      });

    svg.call(zoom as any);

    return () => {
      tooltip.remove();
      svg.selectAll('*').remove();
    };
  };

  return (
    <div className="chart-container">
      <svg ref={svgRef} width={width} height={height}></svg>
    </div>
  );
}
