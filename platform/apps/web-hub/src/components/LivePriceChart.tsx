import React, { useEffect, useRef } from 'react';
import { useWebSocketStore } from '../stores/websocketStore';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface LivePriceChartProps {
  instrumentIds?: string[];
  timeWindow?: number; // minutes to show
  height?: number;
}

export default function LivePriceChart({
  instrumentIds = ['MISO.HUB.INDIANA', 'PJM.HUB.WEST'],
  timeWindow = 60, // 60 minutes
  height = 300
}: LivePriceChartProps) {
  const { latestPrices, getPriceHistory } = useWebSocketStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw chart
    drawChart(ctx, canvas.width, canvas.height);

  }, [latestPrices, instrumentIds, timeWindow, height]);

  const drawChart = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Draw axes
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw price lines for each instrument
    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'];

    instrumentIds.forEach((instrumentId, index) => {
      const history = getPriceHistory(instrumentId, 100); // Get last 100 points
      if (history.length < 2) return;

      const color = colors[index % colors.length];
      drawPriceLine(ctx, history, color, padding, chartWidth, chartHeight, width, height);
    });

    // Draw legend
    drawLegend(ctx, instrumentIds, colors, width, height);
  };

  const drawPriceLine = (
    ctx: CanvasRenderingContext2D,
    history: any[],
    color: string,
    padding: number,
    chartWidth: number,
    chartHeight: number,
    canvasWidth: number,
    canvasHeight: number
  ) => {
    if (history.length === 0) return;

    // Calculate price range
    const prices = history.map(h => h.value);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice || 1;

    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    history.forEach((point, index) => {
      const x = padding + (index / (history.length - 1)) * chartWidth;
      const y = canvasHeight - padding - ((point.value - minPrice) / priceRange) * chartHeight;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw data points
    ctx.fillStyle = color;
    history.forEach((point, index) => {
      const x = padding + (index / (history.length - 1)) * chartWidth;
      const y = canvasHeight - padding - ((point.value - minPrice) / priceRange) * chartHeight;

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const drawLegend = (
    ctx: CanvasRenderingContext2D,
    instrumentIds: string[],
    colors: string[],
    width: number,
    height: number
  ) => {
    const legendY = 20;
    const itemHeight = 20;

    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';

    instrumentIds.forEach((instrumentId, index) => {
      const y = legendY + index * itemHeight;
      const color = colors[index % colors.length];

      // Draw color box
      ctx.fillStyle = color;
      ctx.fillRect(10, y - 12, 12, 12);

      // Draw text
      ctx.fillStyle = '#6b7280';
      ctx.fillText(instrumentId, 30, y);

      // Draw current price
      const price = useWebSocketStore.getState().latestPrices[instrumentId];
      if (price) {
        ctx.fillStyle = '#1f2937';
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText(`$${price.value.toFixed(2)}`, width - 80, y);
      }
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Live Price Chart</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="w-full border rounded"
            style={{ height: `${height}px` }}
          />

          {/* Chart overlay for current time indicator */}
          <div className="absolute top-2 right-2 text-xs text-muted-foreground">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>

        {/* Chart controls */}
        <div className="flex gap-2 mt-4">
          <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
            15m
          </button>
          <button className="px-3 py-1 text-sm border rounded bg-blue-50 border-blue-200">
            1h
          </button>
          <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
            4h
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
