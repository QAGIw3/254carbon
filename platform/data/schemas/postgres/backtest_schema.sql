-- Backtesting Results Schema

CREATE TABLE IF NOT EXISTS pg.backtest_results (
    backtest_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id     TEXT NOT NULL REFERENCES pg.instrument(instrument_id),
    scenario_id       TEXT NOT NULL REFERENCES pg.scenario(scenario_id),
    forecast_date     DATE NOT NULL,
    evaluation_start  DATE NOT NULL,
    evaluation_end    DATE NOT NULL,
    mape              FLOAT NOT NULL,
    wape              FLOAT NOT NULL,
    rmse              FLOAT NOT NULL,
    mean_error        FLOAT NOT NULL,
    median_error      FLOAT NOT NULL,
    n_observations    INT NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_backtest_instrument ON pg.backtest_results(instrument_id, forecast_date DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_created ON pg.backtest_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_scenario ON pg.backtest_results(scenario_id, forecast_date DESC);

