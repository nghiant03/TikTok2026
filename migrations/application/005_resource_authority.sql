CREATE TABLE IF NOT EXISTS authority_resource_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authority_resource_reservations (
    reservation_id TEXT PRIMARY KEY,
    reservation_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('reserved', 'consumed', 'released')),
    settled_usage_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authority_resource_operations (
    operation_id TEXT PRIMARY KEY,
    reservation_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('reserve', 'consume', 'release', 'reconcile')),
    usage_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (reservation_id) REFERENCES authority_resource_reservations(reservation_id)
);

CREATE TABLE IF NOT EXISTS authority_resource_ledger_runs (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    run_id TEXT NOT NULL
);
