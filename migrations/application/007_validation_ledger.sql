CREATE TABLE IF NOT EXISTS authority_validation_operations (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    repair_attempt INTEGER NOT NULL CHECK (repair_attempt >= 0),
    subject_sha256 TEXT NOT NULL,
    implementation_diff_sha256 TEXT,
    operation_json TEXT NOT NULL,
    subject_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, experiment_id, stage, repair_attempt, subject_sha256)
);

CREATE TABLE IF NOT EXISTS authority_validation_reports (
    report_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    report_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    validation_operation_id TEXT NOT NULL UNIQUE,
    FOREIGN KEY (validation_operation_id) REFERENCES authority_validation_operations(operation_id)
);

CREATE TABLE IF NOT EXISTS authority_validation_blockers (
    blocker_id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    blocker_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    validation_operation_id TEXT NOT NULL,
    FOREIGN KEY (report_id) REFERENCES authority_validation_reports(report_id),
    FOREIGN KEY (validation_operation_id) REFERENCES authority_validation_operations(operation_id)
);

CREATE TABLE IF NOT EXISTS authority_blocker_resolutions (
    resolution_id TEXT PRIMARY KEY,
    blocker_id TEXT NOT NULL UNIQUE,
    report_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    resolution_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    validation_operation_id TEXT NOT NULL,
    FOREIGN KEY (blocker_id) REFERENCES authority_validation_blockers(blocker_id),
    FOREIGN KEY (report_id) REFERENCES authority_validation_reports(report_id),
    FOREIGN KEY (validation_operation_id) REFERENCES authority_validation_operations(operation_id)
);

CREATE INDEX IF NOT EXISTS validation_blockers_experiment
ON authority_validation_blockers (experiment_id, created_at, blocker_id);

CREATE INDEX IF NOT EXISTS validation_reports_experiment
ON authority_validation_reports (experiment_id, created_at, report_id);
