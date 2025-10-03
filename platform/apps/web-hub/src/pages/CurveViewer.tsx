import React, { useState } from 'react';
import ForwardCurve3D from '../components/ForwardCurve3D';
import MarketHeatmap from '../components/MarketHeatmap';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

export default function CurveViewer() {
  const [selectedView, setSelectedView] = useState('3d-curves');

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Advanced Market Visualizations</h1>
        <p className="mt-2 text-gray-600">
          Interactive 3D curves and market heatmaps for comprehensive price analysis.
        </p>
      </div>

      <Tabs value={selectedView} onValueChange={setSelectedView} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="3d-curves">3D Forward Curves</TabsTrigger>
          <TabsTrigger value="market-heatmap">Market Heatmap</TabsTrigger>
        </TabsList>

        <TabsContent value="3d-curves" className="space-y-6">
          <ForwardCurve3D height={500} width={800} />

          <Card>
            <CardHeader>
              <CardTitle>Curve Analysis Tools</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Curve Steepness</h4>
                  <p className="text-sm text-muted-foreground">
                    Measures how much prices increase over time. Higher values indicate more contango.
                  </p>
                  <div className="mt-2 text-2xl font-bold text-blue-600">2.3%</div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Volatility Index</h4>
                  <p className="text-sm text-muted-foreground">
                    30-day rolling volatility measure for price uncertainty.
                  </p>
                  <div className="mt-2 text-2xl font-bold text-orange-600">12.4%</div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Basis Risk</h4>
                  <p className="text-sm text-muted-foreground">
                    Hub-to-node price differential risk measure.
                  </p>
                  <div className="mt-2 text-2xl font-bold text-purple-600">$3.2/MWh</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="market-heatmap" className="space-y-6">
          <MarketHeatmap
            height={500}
            width={800}
          />

          <Card>
            <CardHeader>
              <CardTitle>Regional Market Analysis</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Highest Price Region</h4>
                  <p className="text-sm text-muted-foreground">Currently showing highest average prices</p>
                  <div className="mt-2 text-xl font-bold text-red-600">West ($52.3/MWh)</div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Price Dispersion</h4>
                  <p className="text-sm text-muted-foreground">Standard deviation of regional prices</p>
                  <div className="mt-2 text-xl font-bold text-blue-600">$8.7/MWh</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

