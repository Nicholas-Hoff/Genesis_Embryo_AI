import crash_tracker


def test_crashtracker_serialization_round_trip(tmp_path, monkeypatch):
    tmp_file = tmp_path / "crashes.json"
    monkeypatch.setattr(crash_tracker, "_CACHE_FILE", str(tmp_file))

    tracker = crash_tracker.CrashTracker()
    tracker.record_crash("test_goal", "test_phase", {"info": "sample"})

    data = tracker.to_json()
    restored = crash_tracker.CrashTracker.from_json(data)

    assert restored.crashes == tracker.crashes
