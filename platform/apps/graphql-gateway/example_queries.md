# GraphQL Example Queries

Access GraphQL Playground at: `http://localhost:8007/graphql`

## Get Instruments

```graphql
query GetInstruments {
  instruments(market: "power", product: "lmp", limit: 10) {
    instrumentId
    market
    product
    locationCode
    unit
    currency
  }
}
```

## Get Price Ticks

```graphql
query GetPrices {
  priceTicks(
    instrumentId: "MISO.HUB.INDIANA"
    startTime: "2025-10-01T00:00:00Z"
    endTime: "2025-10-03T23:59:59Z"
    priceType: "mid"
    limit: 100
  ) {
    eventTime
    value
    volume
    source
  }
}
```

## Get Forward Curve

```graphql
query GetCurve {
  forwardCurve(
    instrumentId: "PJM.HUB.WEST"
    asOfDate: "2025-10-03"
    scenarioId: "BASE"
  ) {
    instrumentId
    asOfDate
    scenarioId
    points {
      deliveryStart
      deliveryEnd
      tenorType
      price
      currency
    }
  }
}
```

## Get Scenarios

```graphql
query GetScenarios {
  scenarios(limit: 20) {
    scenarioId
    title
    description
    visibility
    createdBy
    createdAt
  }
}
```

## Get Backtest Results

```graphql
query GetBacktests {
  backtestResults(
    instrumentId: "MISO.HUB.INDIANA"
    scenarioId: "BASE"
    limit: 10
  ) {
    instrumentId
    forecastDate
    mape
    wape
    rmse
    meanError
    nObservations
  }
}
```

## Combined Query

Get multiple resources in one request:

```graphql
query GetMarketData {
  # Get instruments
  powerInstruments: instruments(market: "power", limit: 5) {
    instrumentId
    locationCode
  }
  
  # Get prices for specific instrument
  recentPrices: priceTicks(
    instrumentId: "MISO.HUB.INDIANA"
    startTime: "2025-10-02T00:00:00Z"
    endTime: "2025-10-03T23:59:59Z"
    limit: 24
  ) {
    eventTime
    value
  }
  
  # Get forward curve
  curve: forwardCurve(
    instrumentId: "MISO.HUB.INDIANA"
    asOfDate: "2025-10-03"
  ) {
    points {
      deliveryStart
      price
    }
  }
  
  # Get backtest performance
  backtests: backtestResults(
    instrumentId: "MISO.HUB.INDIANA"
    limit: 5
  ) {
    forecastDate
    mape
    rmse
  }
}
```

## Filtering and Pagination

```graphql
query FilteredInstruments {
  miso: instruments(market: "power", limit: 10) {
    instrumentId
    locationCode
  }
  
  gas: instruments(market: "gas", limit: 10) {
    instrumentId
    locationCode
  }
}
```

## LNG Routing Optimization

```graphql
query LngRouting {
  lngRouting(
    asOf: "2025-10-03"
    exportTerminals: ["Sabine Pass"]
    importTerminals: ["Zeebrugge", "Rotterdam"]
    limit: 10
  ) {
    asOfDate
    metadata
    options {
      routeId
      exportTerminal
      importTerminal
      totalCostUsd
      costPerMmbtuUsd
      isOptimalRoute
    }
  }
}
```

## Coal Transport Cost Breakdown

```graphql
query CoalTransportCosts {
  coalTransportCosts(routeId: "newcastle_to_rotterdam", limit: 6) {
    asOfMonth
    transportMode
    cargoTonnes
    totalCostUsd
    breakdown {
      bunkerCostUsd
      portFeesUsd
      congestionPremiumUsd
      carbonCostUsd
    }
  }
}
```

## Correlation Pairs

```graphql
query CorrelationPairs {
  analytics {
    correlationPairs(
      instruments: ["NG1", "NG2", "NG3"]
      start: "2024-01-01"
      end: "2024-03-01"
      minSamples: 45
      limit: 100
    ) {
      date
      instrument1
      instrument2
      correlation
      sampleCount
    }
  }
}
```

## Correlation Matrix

```graphql
query CorrelationMatrix {
  analytics {
    correlationMatrix(
      instruments: ["NG1", "NG2", "NG3"]
      minSamples: 45
    ) {
      date
      instruments
      coefficients
    }
  }
}
```

## Volatility Surface

```graphql
query VolatilitySurface {
  analytics {
    volatilitySurface(
      instruments: ["NG1", "NG2"]
      start: "2023-10-01"
      end: "2024-03-01"
      limit: 500
    ) {
      asOfDate
      instrumentId
      vol30d
      vol90d
      vol365d
    }
  }
}
```

## Seasonality Decomposition

```graphql
query SeasonalityDecomposition {
  analytics {
    seasonalityDecomposition(
      instrumentId: "NG1"
      method: "stl"
      start: "2023-03-01"
      end: "2024-03-01"
      limit: 180
    ) {
      snapshotDate
      trend
      seasonal
      residual
    }
    seasonalityDecompositionLatest(instrumentId: "NG1", method: "stl") {
      snapshotDate
      trend
      seasonal
      residual
    }
  }
}
```

## Research Queries

```graphql
query ResearchQuery {
  analytics {
    listResearchQueries
    researchQuery(
      input: {
        queryId: LIST_NOTEBOOKS
        params: { status: "published" }
        limit: 25
      }
    ) {
      columns
      values
    }
  }
}
```

## Pipeline Congestion Forecast

```graphql
query PipelineForecast {
  pipelineCongestion(pipelineId: "TCO_MAINLINE", limit: 14) {
    forecastDate
    utilizationForecastPct
    riskTier
    riskScore
  }
}
```

## Pipeline Alerts

```graphql
query PipelineAlerts {
  pipelineAlerts(pipelineId: "TCO_MAINLINE", lookaheadDays: 7) {
    date
    utilizationForecastPct
    riskTier
    alertLevel
    message
  }
}
```

## Seasonal Demand Forecast

```graphql
query SeasonalDemandForecast {
  seasonalDemand(
    region: "northeast"
    scenarioId: "BASE"
    limit: 30
  ) {
    region
    scenarioId
    peakAssessment {
      forecastPeakMw
      averagePeakRisk
      observations
    }
    points {
      forecastDate
      finalForecastMw
      peakRiskScore
      confidenceLowMw
      confidenceHighMw
    }
  }
}
```

## Advantages of GraphQL

1. **Request exactly what you need** - no overfetching or underfetching
2. **Single request for multiple resources** - reduced network round trips
3. **Strongly typed schema** - automatic documentation and validation
4. **Introspection** - explore API interactively
5. **Great tooling** - GraphQL Playground, Apollo Client, etc.

## Carbon Price Forecasts

```graphql
query CarbonForecasts {
  carbonPriceForecasts(market: "eua", horizonDays: 365, limit: 30) {
    market
    forecastDate
    forecastPrice
    std
    drivers
    modelVersion
  }
}
```

## Compliance Costs

```graphql
query ComplianceCosts {
  complianceCosts(market: "eua", start: "2025-01-01", limit: 10) {
    asOfDate
    sector
    totalEmissions
    costPerTonne
    totalComplianceCost
    details
  }
}
```

## Decarbonization Pathways

```graphql
query Pathways {
  decarbonizationPathways(sector: "power", scenario: "ambitious", limit: 5) {
    asOfDate
    targetYear
    annualReductionRate
    targetAchieved
    emissionsTrajectory
    technologyAnalysis
  }
}
```

## Renewable Adoption Forecast

```graphql
query RenewableAdoption {
  renewableAdoption(technology: "solar", limit: 10) {
    asOfDate
    forecastYear
    capacityGw
    policySupport
    economicMultipliers
  }
}
```

## Stranded Asset Risk

```graphql
query StrandedAssets {
  strandedAssetRisk(assetType: "coal_generation", limit: 5) {
    asOfDate
    assetValue
    strandedValue
    strandedRatio
    riskLevel
    details
  }
}
```

## Policy Scenario Impacts

```graphql
query PolicyImpacts {
  policyScenarioImpacts(scenario: "HighPolicyTightening", limit: 20) {
    asOfDate
    scenario
    entity
    metric
    value
    details
  }
}
```
