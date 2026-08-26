CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    parent_experiment_id TEXT,
    status TEXT NOT NULL,
    source_commit TEXT,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE audit_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    experiment_id TEXT,
    event_type TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE scientific_lessons (
    lesson_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    evidence_strength TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE artifact_records (
    artifact_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
