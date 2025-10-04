# Infrastructure GraphQL Example Queries

## Get Infrastructure Assets

```graphql
query GetInfrastructureAssets {
  infrastructureAssets(
    assetType: "power_plant"
    country: "US"
    status: "operational"
    limit: 20
  ) {
    assetId
    assetName
    assetType
    country
    latitude
    longitude
    status
    operator
    commissionedDate
  }
}
```

## Get Power Plants by Fuel Type

```graphql
query GetPowerPlants {
  powerPlants(
    country: "DE"
    fuelType: "wind"
    minCapacityMw: 100
    limit: 10
  ) {
    assetId
    assetName
    country
    capacityMw
    primaryFuel
    capacityFactor
    annualGenerationGwh
    latitude
    longitude
  }
}
```

## Get LNG Terminals

```graphql
query GetLNGTerminals {
  lngTerminals(
    country: "ES"
    minCapacityGwh: 100
    limit: 10
  ) {
    assetId
    assetName
    country
    storageCapacityGwh
    regasificationCapacityGwhD
    numTanks
    operator
    latitude
    longitude
  }
}
```

## Get LNG Inventory Time Series

```graphql
query GetLNGInventory {
  lngInventory(
    terminalId: "ES_BARCELONA"
    startTime: "2025-09-01T00:00:00Z"
    endTime: "2025-10-01T00:00:00Z"
    limit: 100
  ) {
    ts
    terminalId
    terminalName
    country
    inventoryGwh
    fullnessPct
    sendOutGwh
    shipArrivals
  }
}
```

## Get Power Generation Data

```graphql
query GetPowerGeneration {
  powerGeneration(
    fuelType: "solar"
    country: "US"
    startTime: "2025-10-01T00:00:00Z"
    endTime: "2025-10-02T00:00:00Z"
    limit: 1000
  ) {
    ts
    plantId
    plantName
    fuelType
    capacityMw
    generationMwh
    capacityFactor
    emissionsTco2
  }
}
```

## Get Renewable Resources

```graphql
query GetRenewableResources {
  renewableResources(
    resourceType: "solar_ghi"
    minLatitude: 40.0
    maxLatitude: 50.0
    minLongitude: -10.0
    maxLongitude: 10.0
    limit: 100
  ) {
    locationId
    latitude
    longitude
    resourceType
    annualAverage
    unit
    dataYear
    monthlyAverages
  }
}
```

## Get Infrastructure Projects

```graphql
query GetInfrastructureProjects {
  infrastructureProjects(
    projectType: "transmission_line"
    status: "construction"
    minCapacityMw: 1000
    limit: 20
  ) {
    projectId
    projectName
    projectType
    countries
    status
    capacityMw
    voltageKv
    lengthKm
    progressPct
    estimatedCostMusd
    startYear
    completionYear
    developer
  }
}
```

## Get Infrastructure Statistics

```graphql
query GetInfrastructureStats {
  infrastructureStats(
    country: "FR"
    assetType: "power_plant"
    startDate: "2025-09-01"
    endDate: "2025-10-01"
  ) {
    date
    country
    assetType
    totalCapacity
    availableCapacity
    numAssets
    numOperational
    avgCapacityFactor
  }
}
```

## Combined Infrastructure Query

Get multiple infrastructure data types in one request:

```graphql
query GetInfrastructureOverview {
  # Power plants in Germany
  germanPowerPlants: powerPlants(country: "DE", limit: 5) {
    assetName
    capacityMw
    primaryFuel
  }
  
  # LNG terminals in Spain
  spanishLngTerminals: lngTerminals(country: "ES", limit: 5) {
    assetName
    storageCapacityGwh
  }
  
  # Active infrastructure projects
  activeProjects: infrastructureProjects(status: "construction", limit: 5) {
    projectName
    countries
    capacityMw
    progressPct
  }
  
  # Infrastructure stats for France
  franceStats: infrastructureStats(
    country: "FR"
    startDate: "2025-10-01"
    endDate: "2025-10-01"
  ) {
    assetType
    totalCapacity
    numOperational
  }
}
```
