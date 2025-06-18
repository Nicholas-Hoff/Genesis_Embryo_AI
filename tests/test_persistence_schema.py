import os
import duckdb
import tempfile
from persistence import MemoryDB


def test_init_tables_adds_missing_columns(tmp_path):
    db_path = tmp_path / "test.db"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE survival_details ("
        "ts TIMESTAMP, heartbeat INT, gene_count INT, "
        "cpu DOUBLE, memory DOUBLE, disk DOUBLE, network DOUBLE, composite DOUBLE)"
    )
    conn.close()

    db = MemoryDB(path=str(db_path))
    db.record_survival_detail(
        1, 1,
        0.1, 0.1, 0.1, 0.1,
        0.5, 10.0, 2.0
    )
    rows = db.conn.execute("SELECT * FROM survival_details").fetchall()
    assert len(rows) == 1
    assert len(rows[0]) == 10
    db.conn.close()
