CREATE TABLE IF NOT EXISTS authority_baseline_calibrations (
    calibration_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authority_run_baseline_bindings (
    run_id TEXT PRIMARY KEY,
    calibration_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (calibration_id) REFERENCES authority_baseline_calibrations(calibration_id)
);

CREATE INDEX IF NOT EXISTS run_baseline_bindings_calibration_lookup
ON authority_run_baseline_bindings (calibration_id);
