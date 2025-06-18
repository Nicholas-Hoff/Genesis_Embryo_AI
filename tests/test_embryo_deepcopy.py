import copy
import pytest

from genesis_embryo_core import Embryo, cfg_mgr


def test_embryo_deepcopy_succeeds(tmp_path):
    pytest.importorskip("torch")
    db_path = tmp_path / "embryo.db"
    embryo = Embryo(cfg_mgr, db_path=str(db_path), disable_snapshots=True)
    clone = copy.deepcopy(embryo)
    assert isinstance(clone, Embryo)
