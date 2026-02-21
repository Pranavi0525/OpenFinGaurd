CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    request_id          VARCHAR(64) UNIQUE,
    decision            VARCHAR(16),
    risk_band           VARCHAR(32),
    default_probability FLOAT,
    model_name          VARCHAR(64),
    model_version       VARCHAR(32),
    inference_ms        FLOAT,
    features_json       JSONB,
    explanation_json    JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_decision ON predictions(decision);
