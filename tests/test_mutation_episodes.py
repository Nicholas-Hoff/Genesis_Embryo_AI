import duckdb
from persistence import MemoryDB, SnapshotManager


def test_record_mutation_episode(tmp_path):
    db_path = tmp_path / "test.db"
    db = MemoryDB(path=str(db_path))
    db.record_mutation_episode("ep1", 1, 2, {"composite": 0.5}, {"composite": 0.6})
    rows = db.conn.execute("SELECT * FROM mutation_episodes").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "ep1"
    db.conn.close()


def test_snapshot_export_mutation_episodes(tmp_path):
    db_path = tmp_path / "test.db"
    db = MemoryDB(path=str(db_path))
    db.record_mutation_episode("ep1", 1, 2, {"composite": 0.5}, {"composite": 0.6})
    snap_dir = tmp_path / "snap"
    snap = SnapshotManager(db_paths=[str(db_path)], snap_dir=str(snap_dir))
    export_dir = snap_dir / "export"
    snap.export_snapshot(str(export_dir))
    snap._q.join()
    snap.shutdown()
    exported = export_dir / "test" / "mutation_episodes.parquet"
    assert exported.exists()
    df = duckdb.sql(f"SELECT * FROM '{exported}'").df()
    assert len(df) == 1

