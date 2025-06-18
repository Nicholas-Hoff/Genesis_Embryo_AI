import types

import pygad_strategy

class DummyEmbryo:
    def __init__(self):
        self.param_bounds = {'x': (0.0, 2.0), 'y': (0.0, 2.0)}
        self.x = 1.0
        self.y = 1.0
        self.mutator = types.SimpleNamespace(params=['x', 'y'])

    def apply_param_bounds(self, param, value):
        low, high = self.param_bounds[param]
        return max(low, min(high, value))

def test_pygad_mutation_alters_params():
    embryo = DummyEmbryo()
    before = (embryo.x, embryo.y)
    desc, ctx = pygad_strategy.pygad_mutation(embryo)
    after = (embryo.x, embryo.y)
    assert after != before
