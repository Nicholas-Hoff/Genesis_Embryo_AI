import ast
import inspect
import random
import time
import logging
from types import FunctionType
from typing import Any, List, Tuple, Callable, Optional, Dict, Union

from strategy import StrategyRegistry, SynthStrategy as BaseSynthStrategy

logger = logging.getLogger(__name__)

class MetaSynthStrategy(BaseSynthStrategy):
    """
    A SynthStrategy that wraps a generated meta-strategy function,
    but treats any NameError, AttributeError, or ZeroDivisionError return as "no-op" rather than fatal.
    """
    def __init__(self, func: FunctionType, name: str):
        super().__init__(name, func)
        self.func = func

    def apply(self, state):
        """
        Call the underlying generated function. If it returns None,
        or if it raises NameError/AttributeError/ZeroDivisionError, treat as a no-op and return ("", {}).
        Any other exception is propagated.
        """
        try:
            result = self.func(state)
        except (NameError, AttributeError, ZeroDivisionError) as e:
            # Skip any generated function that references an invalid attribute or divides by zero
            logger.warning(f"[MetaSynthStrategy] {self.name} skipped due to {type(e).__name__}: {e}")
            return "", {}
        except Exception as e:
            # Any other exception should still be raised
            logger.warning(f"[MetaSynthStrategy] {self.name} failed: {e}")
            raise

        # If the function returned None (i.e. no explicit return), treat as no-op
        if result is None:
            logger.info(f"[MetaSynthStrategy] {self.name} returned None, skipping as no-op")
            return "", {}

        return result

class MetaStrategyEngine:
    """
    Engine for autonomous discovery of new strategies via AST-level
    crossover, mutation, and refinement of existing strategies.
    """
    def __init__(
        self,
        registry: StrategyRegistry,
        batch_size: int = 5,
        interval: float = 60.0,
        mutation_probs: Optional[Dict[str, float]] = None
    ):
        self.registry = registry
        self.batch_size = batch_size
        self._interval = interval
        self._last_gen = 0.0
        self._counter = 0
        # Default mutation probabilities
        default_probs = {
            'const': 0.2,
            'binop': 0.1,
            'compare': 0.1,
            'stmt': 0.1,
            'for_unroll': 0.1,
            'while_to_for': 0.1,
            'if_invert': 0.1,
            'try': 0.05,
            'comprehension': 0.05
        }
        self.mutation_probs = default_probs if mutation_probs is None else {**default_probs, **mutation_probs}

    def _get_strategy_asts(self) -> List[Tuple[str, ast.FunctionDef]]:
        asts: List[Tuple[str, ast.FunctionDef]] = []
        for strat in self.registry.get_all():
            try:
                src = inspect.getsource(strat.fn)
                tree = ast.parse(src)
                funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                if funcs:
                    asts.append((strat.name, funcs[0]))
            except (OSError, TypeError, SyntaxError):
                continue
        return asts

    def _mutate_tree(self, func: ast.FunctionDef) -> ast.FunctionDef:
        probs = self.mutation_probs
        class Mutator(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)) and random.random() < probs['const']:
                    factor = 1 + random.uniform(-0.3, 0.3)
                    node.value = type(node.value)(node.value * factor)
                return node

            def visit_BinOp(self, node):
                if random.random() < probs['binop']:
                    node.op = random.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
                return self.generic_visit(node)

            def visit_Compare(self, node):
                if node.ops and random.random() < probs['compare']:
                    node.ops[0] = random.choice([
                        ast.Eq(), ast.NotEq(), ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE()
                    ])
                return self.generic_visit(node)

            def visit_FunctionDef(self, node):
                node = self.generic_visit(node)
                if node.body and random.random() < probs['stmt']:
                    i = random.randrange(len(node.body))
                    if random.random() < 0.5:
                        del node.body[i]
                    else:
                        node.body.insert(i, ast.copy_location(node.body[i], node.body[i]))
                if node.body and random.random() < probs['try']:
                    idx = random.randrange(len(node.body))
                    stmt = node.body[idx]
                    try_node = ast.Try(
                        body=[stmt],
                        handlers=[ast.ExceptHandler(type=None, name=None, body=[ast.Pass()])],
                        orelse=[],
                        finalbody=[]
                    )
                    node.body[idx] = ast.copy_location(try_node, stmt)
                return node

            def visit_For(self, node):
                node = self.generic_visit(node)
                if node.body and random.random() < probs['for_unroll']:
                    node.body += [ast.copy_location(stmt, stmt) for stmt in node.body]
                return node

            def visit_While(self, node):
                node = self.generic_visit(node)
                if node.body and random.random() < probs['while_to_for']:
                    new_node = ast.For(
                        target=ast.Name(id='_', ctx=ast.Store()),
                        iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(value=1)], keywords=[]),
                        body=node.body,
                        orelse=node.orelse
                    )
                    return ast.copy_location(new_node, node)
                return node

            def visit_If(self, node):
                node = self.generic_visit(node)
                if random.random() < probs['if_invert']:
                    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                    node.body, node.orelse = node.orelse or [], node.body
                return node

            def visit_ListComp(self, node):
                node = self.generic_visit(node)
                if random.random() < probs['comprehension']:
                    return ast.copy_location(
                        ast.GeneratorExp(elt=node.elt, generators=node.generators), node)
                return node

            def visit_DictComp(self, node):
                node = self.generic_visit(node)
                if random.random() < probs['comprehension']:
                    node.key, node.value = node.value, node.key
                return node

        mut_tree = Mutator().visit(func)
        ast.fix_missing_locations(mut_tree)
        return mut_tree

    def _crossover_trees(self, f1: ast.FunctionDef, f2: ast.FunctionDef) -> ast.FunctionDef:
        b1, b2 = f1.body, f2.body
        if not b1 or not b2:
            return f1
        cut1 = random.randint(0, len(b1) - 1)
        cut2 = random.randint(0, len(b2) - 1)
        new_body = b1[:cut1] + b2[cut2:]
        child = ast.FunctionDef(
            name=f1.name,
            args=f1.args,
            body=new_body,
            decorator_list=f1.decorator_list,
            returns=f1.returns,
            type_comment=None
        )
        ast.fix_missing_locations(child)
        return child

    def _compile_strategy(self, func_def: ast.FunctionDef, base_name: str) -> Optional[MetaSynthStrategy]:
        if not func_def.body:
            func_def.body = [ast.Pass()]
        try:
            header = (
                "import random\n"
                "param = ''\n"
                "delta = 0.0\n"
                "old = 0.0\n"
                "new = 0.0\n"
                "desc = ''\n"
                "ctx = {}\n"
                "from typing import Any, Optional, Dict, List, Tuple, Callable, Union\n"
                "from mutation import Archive, MutationEngine, mutation_cycle, DEFAULT_STRATEGIES_MAP, tweak_task_param, TASK_PARAM_SAMPLERS, embryo_mutation\n"
                "from goals import Goal, RollingStats, GoalGenerator, GoalEngine, default_q_values\n"
                "from strategy import StrategyRegistry, SynthStrategy\n"
            )
            name = f"meta_{base_name}_{int(time.time())}_{self._counter}"
            self._counter += 1
            func_def.name = name
            header_module = ast.parse(header)
            module = ast.Module(body=header_module.body + [func_def], type_ignores=[])
            ast.fix_missing_locations(module)
            code = compile(module, "<ast>", "exec")
            exec_ns: Dict[str, Callable] = {}
            exec(code, exec_ns)
            func = exec_ns.get(name)
            if not callable(func):
                return None
            return MetaSynthStrategy(func, name)
        except Exception as e:
            logger.warning(f"[MetaStrategyEngine] compile failed: {e}")
            return None

    def generate_and_register(self):
        now = time.time()
        if now - self._last_gen < self._interval:
            return
        self._last_gen = now
        asts = self._get_strategy_asts()
        if len(asts) < 2:
            return
        new_strats: List[MetaSynthStrategy] = []
        for _ in range(self.batch_size):
            (n1, t1), (n2, t2) = random.sample(asts, 2)
            child = self._crossover_trees(t1, t2)
            child = self._mutate_tree(child)
            strat = self._compile_strategy(child, f"{n1}_{n2}")
            if strat:
                new_strats.append(strat)
        for strat in new_strats:
            self.registry.register(strat)

    def schedule_into(self, embryo):
        embryo.meta_strategy_engine = self
        # the Embryo should call generate_and_register() in each mutate_cycle
