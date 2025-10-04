import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import ForwardCurveSurface from './Three/ForwardCurveSurface';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface CurvePoint {
  delivery_period: string;
  price: number;
  instrument_id: string;
}

interface ForwardCurve3DProps {
  curves?: CurvePoint[][];
  height?: number;
  width?: number;
}

export default function ForwardCurve3D({
  curves = [],
  height = 400,
  width = 600
}: ForwardCurve3DProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || curves.length === 0) return;

    draw3DCurveSurface(svgRef.current, curves, width, height);
  }, [curves, width, height]);

  const draw3DCurveSurface = (
    svg: SVGSVGElement,
    _curveData: CurvePoint[][],
    width: number,
    height: number
  ) => {
    // Clear previous content
    d3.select(svg).selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create main group
    const g = d3.select(svg)
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Sample data for 3D surface visualization
    const sampleData = generateSampleSurfaceData();

    const xScale = d3.scaleLinear()
      .domain(d3.extent(sampleData, d => d.x) as [number, number])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(sampleData, d => d.y) as [number, number])
      .range([innerHeight, 0]);

    const zScale = d3.scaleLinear()
      .domain(d3.extent(sampleData, d => Math.abs(d.z)) as [number, number])
      .range([0, 50]);

    // Create color scale
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain(d3.extent(sampleData, d => d.z) as [number, number]);

    // Draw 3D surface using rectangles for simplicity
    const gridSize = 20;
    const stepSize = 10 / gridSize;

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = i * stepSize;
        const y = j * stepSize;
        const z = Math.sin(x) * Math.cos(y); // Sample function

        // Project 3D coordinates to 2D
        const projectedX = xScale(x + z * 0.3); // Simple perspective
        const projectedY = yScale(y + z * 0.3);
        const size = Math.max(2, zScale(Math.abs(z)) / 10);

        g.append("rect")
          .attr("x", projectedX - size / 2)
          .attr("y", projectedY - size / 2)
          .attr("width", size)
          .attr("height", size)
          .attr("fill", colorScale(z))
          .attr("stroke", "#fff")
          .attr("stroke-width", 0.5)
          .attr("opacity", 0.8);
      }
    }

    // Draw axes
    draw3DAxes(g, innerWidth, innerHeight);

    // Draw labels
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + 30)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text("Delivery Period (Months)");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -30)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text("Price ($/MWh)");

    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .text("3D Forward Curve Surface");
  };

  const draw3DAxes = (
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    width: number,
    height: number
  ) => {
    // Draw X-axis (bottom)
    g.append("line")
      .attr("x1", 0)
      .attr("y1", height)
      .attr("x2", width)
      .attr("y2", height)
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    // Draw Y-axis (left)
    g.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height)
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    // Draw Z-axis (diagonal perspective)
    g.append("line")
      .attr("x1", 0)
      .attr("y1", height)
      .attr("x2", width * 0.3)
      .attr("y2", height * 0.3)
      .attr("stroke", "#333")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");
  };

  const generateSampleSurfaceData = () => {
    // Generate sample 3D surface data for demonstration
    const data = [];
    const steps = 20;

    for (let i = 0; i < steps; i++) {
      for (let j = 0; j < steps; j++) {
        const x = (i / steps) * 10;
        const y = (j / steps) * 10;
        const z = Math.sin(x) * Math.cos(y);

        data.push({ x, y, z });
      }
    }

    return data;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>3D Forward Curve Surface</CardTitle>
      </CardHeader>
      <CardContent>
        <ForwardCurveSurface width={width} height={height} />
      </CardContent>
    </Card>
  );
}
