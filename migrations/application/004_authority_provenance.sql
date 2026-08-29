CREATE TABLE IF NOT EXISTS authority_experiments (
    experiment_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_states (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    transition_id TEXT NOT NULL UNIQUE,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS experiment_states (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    transition_id TEXT NOT NULL UNIQUE,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES authority_experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS source_registrations (
    experiment_id TEXT PRIMARY KEY,
    registration_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES authority_experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS evaluator_identities (
    evaluator_id TEXT PRIMARY KEY,
    evaluator_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authority_artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authority_finalizations (
    finalization_id TEXT PRIMARY KEY,
    finalization_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_reservations (
    reservation_id TEXT PRIMARY KEY,
    reservation_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('reserved', 'consumed', 'released')),
    settled_usage_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_operations (
    operation_id TEXT PRIMARY KEY,
    reservation_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('reserve', 'consume', 'release', 'reconcile')),
    usage_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (reservation_id) REFERENCES resource_reservations(reservation_id)
);

CREATE TABLE IF NOT EXISTS resource_ledger_runs (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    run_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS final_test_claims (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    claim_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS final_test_completions (
    claim_id TEXT PRIMARY KEY,
    finalization_id TEXT NOT NULL UNIQUE,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES final_test_claims(claim_id)
);
