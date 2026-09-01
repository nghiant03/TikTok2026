CREATE TABLE authority_run_experiment_states (
    run_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    status TEXT NOT NULL,
    transition_id TEXT NOT NULL,
    predecessor_transition_id TEXT,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, experiment_id, sequence),
    UNIQUE (run_id, experiment_id, transition_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (experiment_id) REFERENCES authority_experiments(experiment_id)
);

CREATE INDEX run_experiment_states_current
ON authority_run_experiment_states (run_id, experiment_id, sequence DESC);

CREATE INDEX run_experiment_states_status
ON authority_run_experiment_states (run_id, status, sequence, experiment_id);
