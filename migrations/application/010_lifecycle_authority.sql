ALTER TABLE records ADD COLUMN content_sha256 TEXT;
UPDATE records SET content_sha256 = lower(hex(sha256(payload_json)))
WHERE content_sha256 IS NULL;

ALTER TABLE source_registrations ADD COLUMN run_id TEXT;
ALTER TABLE source_registrations ADD COLUMN source_commit TEXT;
ALTER TABLE source_registrations ADD COLUMN eligible INTEGER;
UPDATE source_registrations
SET run_id = json_extract(registration_json, '$.run_id'),
    source_commit = json_extract(registration_json, '$.source_commit'),
    eligible = CASE WHEN json_extract(registration_json, '$.eligible') THEN 1 ELSE 0 END,
    content_sha256 = lower(hex(sha256(registration_json)));

CREATE INDEX IF NOT EXISTS source_registrations_identity
ON source_registrations (registration_id, experiment_id, run_id, source_commit, eligible);

CREATE TABLE IF NOT EXISTS authority_full_attempt_claims (
    attempt_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    source_registration_id TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    attempt_sequence INTEGER NOT NULL CHECK (attempt_sequence BETWEEN 1 AND 50),
    max_attempts INTEGER NOT NULL DEFAULT 50 CHECK (max_attempts = 50),
    attempt_policy_id TEXT NOT NULL DEFAULT 'full-attempt-cap-v1'
        CHECK (attempt_policy_id = 'full-attempt-cap-v1'),
    claim_json TEXT NOT NULL,
    identity_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, attempt_sequence),
    UNIQUE (run_id, attempt_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (experiment_id) REFERENCES authority_experiments(experiment_id),
    FOREIGN KEY (source_registration_id) REFERENCES source_registrations(registration_id)
);

CREATE INDEX IF NOT EXISTS full_attempt_claims_run_order
ON authority_full_attempt_claims (run_id, attempt_sequence, attempt_id);

CREATE INDEX IF NOT EXISTS full_attempt_claims_identity
ON authority_full_attempt_claims (execution_id, attempt_id, run_id, experiment_id,
                                  source_registration_id, source_commit);

CREATE TABLE IF NOT EXISTS authority_scored_observations (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    evaluation_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    observation_json TEXT NOT NULL,
    identity_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    scored_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (attempt_id),
    UNIQUE (evaluation_id),
    FOREIGN KEY (attempt_id) REFERENCES authority_full_attempt_claims(attempt_id),
    FOREIGN KEY (run_id, attempt_id)
        REFERENCES authority_full_attempt_claims(run_id, attempt_id)
);

CREATE INDEX IF NOT EXISTS scored_observations_run_order
ON authority_scored_observations (run_id, attempt_id, observation_id);

CREATE INDEX IF NOT EXISTS scored_observations_identity
ON authority_scored_observations (run_id, experiment_id, attempt_id, evaluation_id,
                                   checkpoint_id, observation_id);

CREATE TABLE IF NOT EXISTS authority_run_closures (
    closure_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    reason TEXT NOT NULL CHECK (reason IN ('plateau', 'attempt_cap')),
    closure_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS run_closures_run_lookup
ON authority_run_closures (run_id, closure_id);

CREATE INDEX IF NOT EXISTS run_closures_identity
ON authority_run_closures (run_id, reason, closure_id);
