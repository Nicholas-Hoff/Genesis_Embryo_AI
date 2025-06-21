import os
import duckdb
from persistence import MemoryDB, SnapshotManager


def test_record_reflection(tmp_path):
    db_path = tmp_path / "test.db"
    db = MemoryDB(path=str(db_path))
    db.record_reflection(
        1, "mode", 2, 0.5, 0.8,
        0.7, "trend", "strat", 0.1
    )
    rows = db.conn.execute("SELECT * FROM reflections").fetchall()
    assert len(rows) == 1
    assert len(rows[0]) == 10
    db.conn.close()


def test_snapshot_export_reflections(tmp_path):
    db_path = tmp_path / "test.db"
    db = MemoryDB(path=str(db_path))
    db.record_reflection(
        1, "mode", 2, 0.5, 0.8,
        0.7, "trend", "strat", 0.1
    )
    snap_dir = tmp_path / "snap"
    snap = SnapshotManager(db_paths=[str(db_path)], snap_dir=str(snap_dir))
    export_dir = snap_dir / "export"
    snap.export_snapshot(str(export_dir))
    snap._q.join()
    snap.shutdown()
    exported = export_dir / "test" / "reflections.parquet"
    assert exported.exists()
    df = duckdb.sql(f"SELECT * FROM '{exported}'").df()
    assert len(df) == 1

