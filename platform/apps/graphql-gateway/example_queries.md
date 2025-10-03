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

## Advantages of GraphQL

1. **Request exactly what you need** - no overfetching or underfetching
2. **Single request for multiple resources** - reduced network round trips
3. **Strongly typed schema** - automatic documentation and validation
4. **Introspection** - explore API interactively
5. **Great tooling** - GraphQL Playground, Apollo Client, etc.

