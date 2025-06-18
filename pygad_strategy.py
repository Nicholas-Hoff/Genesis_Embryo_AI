"""PyGAD-based mutation strategy."""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple, List

import numpy as np
import pygad


def pygad_mutation(embryo: Any) -> Tuple[str, Dict[str, Any]]:
    """Run a tiny GA to evolve a few embryo parameters."""
    # Choose up to 3 parameters to evolve
    params: List[str] = random.sample(embryo.mutator.params, k=min(3, len(embryo.mutator.params)))

    gene_space = []
    start_vals = []
    for p in params:
        lower, upper = embryo.param_bounds[p]
        gene_space.append({'low': lower, 'high': upper})
        start_vals.append(getattr(embryo, p))

    def fitness_func(ga_inst, solution, sol_idx):
        # simple objective: maximize sum of values
        return float(np.sum(solution))

    ga = pygad.GA(
        num_generations=3,
        sol_per_pop=5,
        num_parents_mating=2,
        num_genes=len(params),
        gene_space=gene_space,
        fitness_func=fitness_func,
        mutation_percent_genes=50,
    )
    ga.run()
    best_solution, _, _ = ga.best_solution()

    desc_parts = []
    ctx = {
        'strategy': 'pygad',
        'param': [],
        'old': [],
        'new': [],
        'delta': []
    }
    for val, p, old in zip(best_solution, params, start_vals):
        new_val = embryo.apply_param_bounds(p, val)
        setattr(embryo, p, new_val)
        desc_parts.append(f"{p}: {old:.3f} -> {new_val:.3f}")
        ctx['param'].append(p)
        ctx['old'].append(round(old, 4))
        ctx['new'].append(round(new_val, 4))
        ctx['delta'].append(round(new_val - old, 4))

    desc = '; '.join(desc_parts)
    return desc, ctx

__all__ = ['pygad_mutation']
