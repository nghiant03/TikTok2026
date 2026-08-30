ALTER TABLE source_registrations RENAME TO source_registrations_legacy;

CREATE TABLE source_registrations (
    registration_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    registration_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (experiment_id, revision),
    FOREIGN KEY (experiment_id) REFERENCES authority_experiments(experiment_id)
);

INSERT INTO source_registrations (
    registration_id,
    experiment_id,
    revision,
    registration_json,
    content_sha256,
    created_at
)
SELECT
    'source-' || json_extract(registration_json, '$.source_commit'),
    experiment_id,
    0,
    registration_json,
    content_sha256,
    created_at
FROM source_registrations_legacy;

DROP TABLE source_registrations_legacy;

CREATE INDEX source_registrations_experiment_revision
ON source_registrations (experiment_id, revision DESC);
