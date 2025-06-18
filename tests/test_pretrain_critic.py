import pytest

torch = pytest.importorskip("torch")

import duckdb

from pretrain_critic import load_transitions, pretrain_critic
from genesis_embryo_core import Critic


def build_temp_db(path):
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE transitions (
            hb INTEGER,
            survival_before FLOAT, novelty_before FLOAT, efficiency_before FLOAT,
            mutation_error_before FLOAT, cycle_before FLOAT,
            reward FLOAT,
            survival_after FLOAT, novelty_after FLOAT, efficiency_after FLOAT,
            mutation_error_after FLOAT, cycle_after FLOAT,
            ts INTEGER
        )
        """
    )
    rows = [
        (1, 0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 0.2, 0.3, 0.4, 0.5, 2.0, 1000),
        (2, 0.2, 0.3, 0.4, 0.5, 2.0, 0.8, 0.3, 0.4, 0.5, 0.6, 3.0, 1001),
    ]
    con.executemany(
        "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()



def test_load_transitions(tmp_path):
    db_file = tmp_path / "t.db"
    build_temp_db(db_file)

    df = load_transitions(str(db_file))
    assert len(df) == 2
    assert df.iloc[0]["state"] == [0.1, 0.2, 0.3, 0.4, 1.0]
    assert df.iloc[0]["next_state"] == [0.2, 0.3, 0.4, 0.5, 2.0]
    assert df.iloc[0]["reward"] == pytest.approx(1.0)


def test_pretrain_updates_params(tmp_path):
    db_file = tmp_path / "t.db"
    build_temp_db(db_file)
    df = load_transitions(str(db_file))

    critic = Critic(input_dim=5, hidden=4)
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)

    before = [p.clone() for p in critic.parameters()]
    pretrain_critic(df, critic, optimizer, epochs=1, batch_size=2, device="cpu")
    after = list(critic.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_pretrain_handles_float64_input(tmp_path):
    db_file = tmp_path / "t.db"
    build_temp_db(db_file)
    df = load_transitions(str(db_file))

    # convert numeric data to float64 to mimic higher precision input
    df["reward"] = df["reward"].astype("float64")
    df["state"] = df["state"].apply(lambda x: [float(v) for v in x])
    df["next_state"] = df["next_state"].apply(lambda x: [float(v) for v in x])

    critic = Critic(input_dim=5, hidden=4)
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)

    before = [p.clone() for p in critic.parameters()]
    pretrain_critic(df, critic, optimizer, epochs=1, batch_size=2, device="cpu")
    after = list(critic.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed
