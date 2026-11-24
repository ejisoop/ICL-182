"""Deterministic generator for the paired in-context command-learning task.

A *case* consists of

* a query program (``n_init`` variable initialisations + ``n_stmts`` update
  statements, exactly one of which uses the target made-up command),
* a target command and two distractor made-up commands,
* two different meanings (A, B) for the target command, chosen such that the
  queried variable ends with different values under A and B,
* ``k_max`` demonstrations per (command, meaning), each individually
  unambiguous (it singles out its command's true operation among the four),
* a fixed interleaving order of demonstration slots.

From a case we can render, for any demonstration count k <= k_max, the two
prompts of a *pair* (meaning A / meaning B).  The two prompts share the query
program, initialisation, query target and the distractor demonstrations; they
differ only in the target-command demonstrations.  For the direct-definition
control we render ``M <cmd> means <op> .`` statements instead of demonstrations.

Query-program *templates* (target position, sequence of built-in ops, data flow) are hashed deterministically into train / val / test buckets so that test
query programs have structures never seen during training.

All answers are computed by :mod:`src.data.interpreter` on the rendered text.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from . import interpreter
from .tokenizer import COMMANDS, MODULUS, OPS, VARIABLES

N_COMMANDS_PER_PROMPT = 3  # 1 target + 2 distractors
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass(frozen=True)
class TaskConfig:
    """Query-program complexity (fixed within an experiment)."""
    n_init: int = 2    # number of variable initialisations
    n_stmts: int = 2   # number of update statements (exactly one uses the target command)
    # Experiment 2: every demonstration is consistent with exactly two operations (the true
    # one and a uniformly chosen "confuser"), so one demonstration never identifies the
    # meaning and evidence must be combined across demonstrations.
    ambiguous_demos: bool = False


# ---------------------------------------------------------------------------
# templates and splits
# ---------------------------------------------------------------------------
# A template fixes the *structure* of a query program: the position of the
# target-command statement, the built-in op of every other statement, and the
# data flow of every statement (which variable slot it updates and whether its
# argument is a literal or which variable slot it reads).  Concrete variable
# names, initial values and literal arguments are sampled freely.
Template = Tuple[int, Tuple[str, ...], Tuple[Tuple[int, str], ...]]


def template_key(t: Template) -> str:
    pos, ops, flow = t
    return f"{pos}|{','.join(ops)}|{','.join(f'{v}:{a}' for v, a in flow)}"


def template_split(t: Template) -> str:
    """Deterministic, seed-independent split assignment of a template."""
    h = int(hashlib.md5(template_key(t).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < SPLIT_FRACTIONS["train"]:
        return "train"
    if h < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


def _stmt_choices(task: TaskConfig):
    args = ["n"] + [f"v{j}" for j in range(task.n_init)]
    return [(v, a) for v in range(task.n_init) for a in args]


def all_templates(task: TaskConfig = TaskConfig()) -> List[Template]:
    import itertools
    out = []
    for pos in range(task.n_stmts):
        for ops in itertools.product(OPS, repeat=task.n_stmts - 1):
            for flow in itertools.product(_stmt_choices(task), repeat=task.n_stmts):
                out.append((pos, ops, flow))
    return out


def sample_template(rng: random.Random, split: str, task: TaskConfig = TaskConfig()) -> Template:
    choices = _stmt_choices(task)
    while True:
        t = (rng.randrange(task.n_stmts),
             tuple(rng.choice(OPS) for _ in range(task.n_stmts - 1)),
             tuple(rng.choice(choices) for _ in range(task.n_stmts)))
        if template_split(t) == split:
            return t


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------
@dataclass
class Demo:
    var: str
    init: int
    arg: int

    def render(self, cmd: str, result: int) -> str:
        return f"D {self.var} = {self.init} ; {cmd} {self.var} {self.arg} ; {self.var} -> {result} ."


@dataclass
class Case:
    case_id: str
    template: str
    program: str               # query program text, e.g. "a = 2 ; ... ; sub a 1 ;"
    query_var: str
    target: str                # target made-up command
    distractors: List[str]
    meaning_a: str
    meaning_b: str
    distractor_meanings: List[str]
    demos_a: List[dict]        # k_max demos for target under meaning A
    demos_b: List[dict]        # k_max demos for target under meaning B
    demos_distractor: List[List[dict]]
    slot_order: List[Tuple[int, int]]  # (command slot 0=target/1../, demo index)
    def_order: List[int]       # order of command slots in direct-definition prompts
    answer_a: int = -1
    answer_b: int = -1

    # -- rendering ---------------------------------------------------------
    def _query_text(self) -> str:
        return f"Q {self.program} {self.query_var} ->"

    def meanings(self, which: str) -> Dict[str, str]:
        m = {self.target: self.meaning_a if which == "a" else self.meaning_b}
        m.update(dict(zip(self.distractors, self.distractor_meanings)))
        return m

    def render_icl(self, which: str, k: int) -> str:
        """Prompt with k demonstrations per command (no answer appended)."""
        cmds = [self.target] + self.distractors
        meanings = self.meanings(which)
        tdemos = self.demos_a if which == "a" else self.demos_b
        demo_lists = [tdemos] + self.demos_distractor
        parts = ["<bos>"]
        for slot, idx in self.slot_order:
            if idx >= k:
                continue
            d = Demo(**demo_lists[slot][idx])
            cmd = cmds[slot]
            res = interpreter.query(f"{d.var} = {d.init} ; {cmd} {d.var} {d.arg}", d.var,
                                    {cmd: meanings[cmd]})
            parts.append(d.render(cmd, res))
        parts.append(self._query_text())
        return " ".join(parts)

    def render_definition(self, which: str) -> str:
        cmds = [self.target] + self.distractors
        meanings = self.meanings(which)
        parts = ["<bos>"]
        for slot in self.def_order:
            parts.append(f"M {cmds[slot]} means {meanings[cmds[slot]]} .")
        parts.append(self._query_text())
        return " ".join(parts)

    def answer(self, which: str) -> int:
        return self.answer_a if which == "a" else self.answer_b


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------
def _demo_is_unambiguous(d: Demo, op: str) -> bool:
    """True iff the demo's outcome under ``op`` differs from all other ops."""
    prog = f"{d.var} = {d.init} ; X {d.var} {d.arg}"
    true = interpreter.query(prog, d.var, {"X": op})
    return all(interpreter.query(prog, d.var, {"X": o}) != true for o in OPS if o != op)


def _consistent_ops(init: int, arg: int) -> dict:
    prog = f"a = {init} ; X a {arg}"
    res = {o: interpreter.query(prog, "a", {"X": o}) for o in OPS}
    return res


_AMBIG_TABLE: dict | None = None


def ambiguous_table() -> dict:
    """{(op, confuser): [(init, arg), ...]} with consistent-op set exactly {op, confuser}."""
    global _AMBIG_TABLE
    if _AMBIG_TABLE is None:
        t = {}
        for x in range(MODULUS):
            for n in range(MODULUS):
                res = _consistent_ops(x, n)
                for o in OPS:
                    same = sorted(p for p in OPS if res[p] == res[o])
                    if len(same) == 2:
                        conf = same[0] if same[1] == o else same[1]
                        t.setdefault((o, conf), []).append((x, n))
        _AMBIG_TABLE = t
    return _AMBIG_TABLE


def sample_ambiguous_demo(rng: random.Random, op: str) -> Demo:
    conf = rng.choice([o for o in OPS if o != op])
    x, n = rng.choice(ambiguous_table()[(op, conf)])
    return Demo(var=rng.choice(VARIABLES), init=x, arg=n)


def sample_demo(rng: random.Random, op: str, ambiguous: bool = False) -> Demo:
    if ambiguous:
        return sample_ambiguous_demo(rng, op)
    while True:
        d = Demo(var=rng.choice(VARIABLES), init=rng.randrange(MODULUS),
                 arg=rng.randrange(MODULUS))
        if _demo_is_unambiguous(d, op):
            return d


def _render_program(template: Template, vars_: Sequence[str], values: Sequence[int],
                    rng: random.Random, target: str) -> str:
    pos, ops, flow = template
    stmts = [f"{v} = {x}" for v, x in zip(vars_, values)]
    op_iter = iter(ops)
    for i, (vslot, aspec) in enumerate(flow):
        name = target if i == pos else next(op_iter)
        arg = str(rng.randrange(MODULUS)) if aspec == "n" else vars_[int(aspec[1:])]
        stmts.append(f"{name} {vars_[vslot]} {arg}")
    return " ; ".join(stmts) + " ;"


def sample_case(rng: random.Random, split: str, k_max: int, case_id: str = "",
                task: TaskConfig = TaskConfig()) -> Case:
    while True:
        template = sample_template(rng, split, task)
        cmds = rng.sample(COMMANDS, N_COMMANDS_PER_PROMPT)
        target, distractors = cmds[0], cmds[1:]
        meaning_a, meaning_b = rng.sample(OPS, 2)
        dmeanings = [rng.choice(OPS) for _ in distractors]
        vars_ = rng.sample(VARIABLES, task.n_init)
        values = [rng.randrange(MODULUS) for _ in vars_]
        program = _render_program(template, vars_, values, rng, target)
        env_a = interpreter.execute(program, {target: meaning_a})
        env_b = interpreter.execute(program, {target: meaning_b})
        cands = [v for v in vars_ if env_a[v] != env_b[v]]
        if not cands:
            continue  # query program does not distinguish the two meanings
        qvar = rng.choice(cands)
        amb = task.ambiguous_demos
        demos_a = [asdict(sample_demo(rng, meaning_a, amb)) for _ in range(k_max)]
        demos_b = [asdict(sample_demo(rng, meaning_b, amb)) for _ in range(k_max)]
        demos_d = [[asdict(sample_demo(rng, m, amb)) for _ in range(k_max)] for m in dmeanings]
        slots = [(s, i) for s in range(N_COMMANDS_PER_PROMPT) for i in range(k_max)]
        rng.shuffle(slots)
        def_order = list(range(N_COMMANDS_PER_PROMPT))
        rng.shuffle(def_order)
        case = Case(case_id=case_id, template=template_key(template), program=program,
                    query_var=qvar, target=target, distractors=distractors,
                    meaning_a=meaning_a, meaning_b=meaning_b, distractor_meanings=dmeanings,
                    demos_a=demos_a, demos_b=demos_b, demos_distractor=demos_d,
                    slot_order=slots, def_order=def_order)
        # ground truth from the independent interpreter on the rendered prompt
        case.answer_a = interpreter.answer_prompt(case.render_definition("a"))
        case.answer_b = interpreter.answer_prompt(case.render_definition("b"))
        assert case.answer_a == env_a[qvar] and case.answer_b == env_b[qvar]
        return case


def case_seed(base_seed: int, split: str, index: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{split}|{index}".encode()).hexdigest()
    return int(h[:16], 16)


# ---------------------------------------------------------------------------
# training samples
# ---------------------------------------------------------------------------
@dataclass
class TrainSampleConfig:
    k_min: int = 0
    k_max: int = 8
    p_definition: float = 0.25   # fraction of direct-definition prompts
    n_init: int = 2
    n_stmts: int = 2
    ambiguous_demos: bool = False

    @property
    def task(self) -> TaskConfig:
        return TaskConfig(self.n_init, self.n_stmts, self.ambiguous_demos)


def sample_training_prompt(seed: int, cfg: TrainSampleConfig, split: str = "train") -> Tuple[str, dict]:
    """One training prompt (text including answer and <eos>) + metadata.

    A training prompt is one side (A or B, chosen at random) of a freshly
    generated case drawn from ``split`` templates, so that the training
    distribution matches the evaluation distribution except for the held-out
    template structures."""
    rng = random.Random(seed)
    case = sample_case(rng, split, cfg.k_max, task=cfg.task)
    which = rng.choice("ab")
    if rng.random() < cfg.p_definition:
        prompt, fmt, k = case.render_definition(which), "definition", -1
    else:
        k = rng.randint(cfg.k_min, cfg.k_max)
        prompt, fmt = case.render_icl(which, k), "icl"
    ans = case.answer(which)
    meta = {"format": fmt, "k": k, "op": case.meaning_a if which == "a" else case.meaning_b,
            "template": case.template, "program": case.program}
    return f"{prompt} {ans} <eos>", meta
