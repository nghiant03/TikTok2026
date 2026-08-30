ALTER TABLE authority_validation_blockers ADD COLUMN criterion_id TEXT;

UPDATE authority_validation_blockers
SET criterion_id = json_extract(blocker_json, '$.criterion_id')
WHERE json_extract(blocker_json, '$.criterion_id') IS NOT NULL;

ALTER TABLE authority_blocker_resolutions ADD COLUMN criterion_id TEXT;
ALTER TABLE authority_blocker_resolutions ADD COLUMN status TEXT;

UPDATE authority_blocker_resolutions
SET criterion_id = json_extract(resolution_json, '$.criterion_id'),
    status = json_extract(resolution_json, '$.status')
WHERE json_extract(resolution_json, '$.criterion_id') IS NOT NULL;

-- A stable criterion can be resolved, reopened, and resolved again.  Keep
-- every resolution event instead of enforcing one lifetime resolution per
-- blocker; resolution_id remains the idempotency key.
CREATE TABLE authority_blocker_resolutions_append_only (
    resolution_id TEXT PRIMARY KEY,
    blocker_id TEXT NOT NULL,
    report_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    resolution_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    validation_operation_id TEXT NOT NULL,
    criterion_id TEXT,
    status TEXT,
    FOREIGN KEY (blocker_id) REFERENCES authority_validation_blockers(blocker_id),
    FOREIGN KEY (report_id) REFERENCES authority_validation_reports(report_id),
    FOREIGN KEY (validation_operation_id) REFERENCES authority_validation_operations(operation_id)
);

INSERT INTO authority_blocker_resolutions_append_only
SELECT resolution_id, blocker_id, report_id, experiment_id, resolution_json,
       content_sha256, created_at, validation_operation_id, criterion_id, status
FROM authority_blocker_resolutions;

DROP TABLE authority_blocker_resolutions;
ALTER TABLE authority_blocker_resolutions_append_only RENAME TO authority_blocker_resolutions;

CREATE TABLE IF NOT EXISTS authority_validation_criterion_occurrences (
    occurrence_id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    criterion_id TEXT NOT NULL,
    status TEXT NOT NULL,
    assessment_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (report_id, criterion_id),
    FOREIGN KEY (report_id) REFERENCES authority_validation_reports(report_id)
);

CREATE TABLE IF NOT EXISTS authority_validation_criterion_occurrence_blockers (
    occurrence_id TEXT NOT NULL,
    blocker_id TEXT NOT NULL,
    PRIMARY KEY (occurrence_id, blocker_id),
    FOREIGN KEY (occurrence_id)
        REFERENCES authority_validation_criterion_occurrences(occurrence_id),
    FOREIGN KEY (blocker_id) REFERENCES authority_validation_blockers(blocker_id)
);

CREATE TABLE IF NOT EXISTS authority_validation_resolution_claims (
    claim_id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    criterion_id TEXT NOT NULL,
    status TEXT NOT NULL,
    claim_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (report_id, criterion_id),
    FOREIGN KEY (report_id) REFERENCES authority_validation_reports(report_id)
);

CREATE INDEX IF NOT EXISTS validation_criterion_occurrences_lookup
ON authority_validation_criterion_occurrences (experiment_id, criterion_id, status);

CREATE INDEX IF NOT EXISTS validation_occurrence_blockers_lookup
ON authority_validation_criterion_occurrence_blockers (blocker_id);

CREATE INDEX IF NOT EXISTS validation_resolution_claims_lookup
ON authority_validation_resolution_claims (experiment_id, criterion_id, created_at);

CREATE INDEX IF NOT EXISTS validation_blockers_criterion_lookup
ON authority_validation_blockers (experiment_id, stage, criterion_id);
