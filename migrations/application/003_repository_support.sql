CREATE TABLE IF NOT EXISTS final_test_access (
    run_id TEXT PRIMARY KEY,
    claimed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS records (
    kind TEXT NOT NULL,
    record_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (kind, record_id)
);
