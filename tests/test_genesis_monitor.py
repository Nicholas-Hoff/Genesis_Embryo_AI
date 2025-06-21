import pandas as pd
import pytest
import duckdb

import genesis_monitor as GM


def test_load_survival_details_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(GM, "SURVIVAL_DETAILS_FILE", str(tmp_path / "missing.parquet"))
    df = GM.load_survival_details()
    assert df.empty
    assert list(df.columns) == [
        'ts', 'heartbeat', 'gene_count',
        'cpu', 'memory', 'disk',
        'network', 'composite',
        'cpu_pct', 'disk_io'
    ]


def test_load_survival_details_reads_parquet(monkeypatch, tmp_path):
    path = tmp_path / "survival.parquet"
    exp = pd.DataFrame({
        'ts': pd.to_datetime([1], unit='s'),
        'heartbeat': [1],
        'gene_count': [2],
        'cpu': [0.1],
        'memory': [0.2],
        'disk': [0.3],
        'network': [0.4],
        'composite': [0.5],
        'cpu_pct': [0.6],
        'disk_io': [0.7]
    })
    con = duckdb.connect()
    con.register('exp', exp)
    con.execute(f"COPY exp TO '{path}' (FORMAT PARQUET)")
    con.close()
    monkeypatch.setattr(GM, "SURVIVAL_DETAILS_FILE", str(path))
    df = GM.load_survival_details()
    pd.testing.assert_frame_equal(df, exp)


def test_load_mutation_episodes_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(GM, "MUTATION_EPISODES_FILE", str(tmp_path / "missing.parquet"))
    df = GM.load_mutation_episodes()
    assert df.empty
    assert list(df.columns) == [
        'ts', 'episode_id', 'strategies_applied', 'parameters_changed',
        'survival_before', 'survival_after', 'survival_change'
    ]


def test_load_mutation_episodes_reads_parquet(monkeypatch, tmp_path):
    path = tmp_path / "episodes.parquet"
    exp = pd.DataFrame({
        'ts': pd.to_datetime([1], unit='s'),
        'episode_id': ['1'],
        'strategies_applied': [1],
        'parameters_changed': [2],
        'survival_before': [0.5],
        'survival_after': [0.6],
        'survival_change': [0.1]
    })
    con = duckdb.connect()
    con.register('exp', exp)
    con.execute(f"COPY exp TO '{path}' (FORMAT PARQUET)")
    con.close()
    monkeypatch.setattr(GM, "MUTATION_EPISODES_FILE", str(path))
    df = GM.load_mutation_episodes()
    pd.testing.assert_frame_equal(df, exp)


def test_update_dashboard_uses_memory(monkeypatch):
    hb = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'survival_score': [0.5], 'heartbeat': [1]})
    mut = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'strategy': ['s'], 'param': ['p'], 'survival_change': [0.1]})
    res = pd.DataFrame({
        'ts': pd.to_datetime([0], unit='s'), 'heartbeat': [1], 'gene_count': [1],
        'cpu': [0.1], 'memory': [0.2], 'disk': [0.3],
        'network': [0.4], 'composite': [0.5],
        'cpu_pct': [0.6], 'disk_io': [0.7]
    })
    ep = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'episode_id': [1], 'strategies_applied': ['s'],
                       'parameters_changed': ['p'], 'survival_before': [0.4],
                       'survival_after': [0.5], 'survival_change': [0.1]})
    refl = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'heartbeat': [1], 'mode': ['m'], 'gene_count': [1],
                         'heartbeat_interval': [1], 'survival_threshold': [0.4],
                         'recent_survival': [0.5], 'trend': [1],
                         'most_used_strategy': ['s'], 'mutation_rate': [0.1]})
    mm = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'target_rate': [0.1], 'observed_rate': [0.1]})
    fatal = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'description': ['oops']})

    monkeypatch.setattr(GM, 'load_heartbeats', lambda: hb)
    monkeypatch.setattr(GM, 'load_mutations', lambda: mut)
    monkeypatch.setattr(GM, 'load_survival_details', lambda: res)
    monkeypatch.setattr(GM, 'load_mutation_episodes', lambda: ep)
    monkeypatch.setattr(GM, 'load_reflections', lambda: refl)
    monkeypatch.setattr(GM, 'load_mutation_metrics', lambda: mm)
    monkeypatch.setattr(GM, 'load_fatal_events', lambda: fatal)

    out = GM.update_dashboard(0, None)
    res_fig = out[7]
    assert res_fig.data[1].name == 'Mem'
    assert list(res_fig.data[1].y) == [0.2]
    parcoords = out[8]
    assert list(parcoords.data[0].dimensions[1].values) == [0.2]


def test_dashboard_layout_contains_heartbeat_graph():
    layout = GM.app.layout
    found = False

    def walk(item):
        nonlocal found
        if isinstance(item, (list, tuple)):
            for c in item:
                walk(c)
        else:
            if getattr(item, 'id', None) == 'heartbeat-graph':
                found = True
            if hasattr(item, 'children'):
                walk(item.children)

    walk(layout.children)
    assert found


def test_update_dashboard_error_with_mem(monkeypatch):
    hb = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'survival_score': [0.5], 'heartbeat': [1]})
    mut = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'strategy': ['s'], 'param': ['p'], 'survival_change': [0.1]})
    res = pd.DataFrame({
        'ts': pd.to_datetime([0], unit='s'), 'heartbeat': [1], 'gene_count': [1],
        'cpu': [0.1], 'mem': [0.2], 'disk': [0.3],
        'network': [0.4], 'composite': [0.5],
        'cpu_pct': [0.6], 'disk_io': [0.7]
    })
    ep = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'episode_id': [1], 'strategies_applied': ['s'],
                       'parameters_changed': ['p'], 'survival_before': [0.4],
                       'survival_after': [0.5], 'survival_change': [0.1]})
    refl = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'heartbeat': [1], 'mode': ['m'], 'gene_count': [1],
                         'heartbeat_interval': [1], 'survival_threshold': [0.4],
                         'recent_survival': [0.5], 'trend': [1],
                         'most_used_strategy': ['s'], 'mutation_rate': [0.1]})
    mm = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'target_rate': [0.1], 'observed_rate': [0.1]})
    fatal = pd.DataFrame({'ts': pd.to_datetime([0], unit='s'), 'description': ['oops']})

    monkeypatch.setattr(GM, 'load_heartbeats', lambda: hb)
    monkeypatch.setattr(GM, 'load_mutations', lambda: mut)
    monkeypatch.setattr(GM, 'load_survival_details', lambda: res)
    monkeypatch.setattr(GM, 'load_mutation_episodes', lambda: ep)
    monkeypatch.setattr(GM, 'load_reflections', lambda: refl)
    monkeypatch.setattr(GM, 'load_mutation_metrics', lambda: mm)
    monkeypatch.setattr(GM, 'load_fatal_events', lambda: fatal)

    with pytest.raises(KeyError):
        GM.update_dashboard(0, None)

