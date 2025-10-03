import React, { useEffect, useRef } from 'react';
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

    drawChart();
  }, [data, width, height, margin, color]);

  const drawChart = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous content

    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Create scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(data, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, chartWidth]);

    const yScale = d3
      .scaleLinear()
      .domain([
        d3.min(data, d => d.price) * 0.95,
        d3.max(data, d => d.price) * 1.05
      ])
      .range([chartHeight, 0]);

    // Create line generator
    const line = d3
      .line<PriceData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.price))
      .curve(d3.curveMonotoneX);

    // Create area generator for confidence bands (if volume data available)
    const area = d3
      .area<PriceData>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(d => yScale((d.price * (1 - (d.volume || 0) / 1000))))
      .y1(d => yScale((d.price * (1 + (d.volume || 0) / 1000))))
      .curve(d3.curveMonotoneX);

    // Create main group
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(
        d3.axisBottom(xScale)
          .tickSize(-chartHeight)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .style('stroke', '#e5e7eb')
      .style('stroke-width', 0.5);

    g.append('g')
      .attr('class', 'grid')
      .call(
        d3.axisLeft(yScale)
          .tickSize(-chartWidth)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .style('stroke', '#e5e7eb')
      .style('stroke-width', 0.5);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .style('text-anchor', 'end');

    // Add area for volume-based confidence bands
    if (data.some(d => d.volume !== undefined)) {
      g.append('path')
        .datum(data)
        .attr('class', 'area')
        .attr('d', area)
        .style('fill', color)
        .style('opacity', 0.1);
    }

    // Add price line
    g.append('path')
      .datum(data)
      .attr('class', 'line')
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', color)
      .style('stroke-width', 2);

    // Add data points
    g.selectAll('.point')
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
    const tooltip = d3.select('body')
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

    g.selectAll('.point')
      .on('mouseover', function(event, d) {
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
      .on('zoom', (event) => {
        const newXScale = event.transform.rescaleX(xScale);
        const newYScale = event.transform.rescaleY(yScale);

        g.select('.line').attr('d', line.x(d => newXScale(new Date(d.timestamp))));
        g.select('.area').attr('d', area.x(d => newXScale(new Date(d.timestamp))));

        g.selectAll('.point')
          .attr('cx', d => newXScale(new Date(d.timestamp)))
          .attr('cy', d => newYScale(d.price));

        g.select('.grid').call(
          d3.axisBottom(newXScale)
            .tickSize(-chartHeight)
            .tickFormat(() => '')
        );

        g.selectAll('.x-axis')
          .call(d3.axisBottom(newXScale).ticks(5));
      });

    svg.call(zoom);
  };

  return (
    <div className="chart-container">
      <svg ref={svgRef} width={width} height={height}></svg>
    </div>
  );
}
