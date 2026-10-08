CREATE TABLE IF NOT EXISTS table_snapshots (
    table_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);
