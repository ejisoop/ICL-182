"""Independent reference interpreter for the task language.

The interpreter works purely on *rendered text* (it re-parses the token
stream) and does not share any code with the generator other than the
modulus constant.  It is used to produce every ground-truth answer.

Language
--------
Statements are separated by ``;``.

    <var> = <num>              initialise / assign a literal
    <op>  <var> <arg>          op in {set, add, sub, mul}; arg is a number or variable
    <cmd> <var> <arg>          cmd is a made-up word whose meaning (one of the
                               ops) is supplied by ``meanings``

All arithmetic is modulo MODULUS.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

from .tokenizer import MODULUS

BUILTIN_OPS = ("set", "add", "sub", "mul")


class InterpreterError(Exception):
    pass


def apply_op(op: str, old: int, arg: int, modulus: int = MODULUS) -> int:
    if op == "set":
        return arg % modulus
    if op == "add":
        return (old + arg) % modulus
    if op == "sub":
        return (old - arg) % modulus
    if op == "mul":
        return (old * arg) % modulus
    raise InterpreterError(f"unknown op {op!r}")


def parse_statements(program: str) -> List[List[str]]:
    stmts = []
    for raw in program.split(";"):
        toks = raw.split()
        if toks:
            stmts.append(toks)
    return stmts


def execute(program: str, meanings: Mapping[str, str] | None = None,
            modulus: int = MODULUS) -> Dict[str, int]:
    """Execute ``program`` and return the final variable environment."""
    meanings = dict(meanings or {})
    env: Dict[str, int] = {}

    def value(tok: str) -> int:
        if tok.isdigit():
            return int(tok) % modulus
        if tok in env:
            return env[tok]
        raise InterpreterError(f"read of undefined variable {tok!r}")

    for toks in parse_statements(program):
        if len(toks) == 3 and toks[1] == "=":
            var, _, num = toks
            if not num.isdigit():
                raise InterpreterError(f"bad assignment {toks}")
            env[var] = int(num) % modulus
        elif len(toks) == 3:
            name, var, arg = toks
            if name in BUILTIN_OPS:
                op = name
            elif name in meanings:
                op = meanings[name]
            else:
                raise InterpreterError(f"command {name!r} has no meaning")
            if var not in env:
                raise InterpreterError(f"update of undefined variable {var!r}")
            env[var] = apply_op(op, env[var], value(arg), modulus)
        else:
            raise InterpreterError(f"cannot parse statement {toks}")
    return env


def query(program: str, var: str, meanings: Mapping[str, str] | None = None) -> int:
    env = execute(program, meanings)
    if var not in env:
        raise InterpreterError(f"query of undefined variable {var!r}")
    return env[var]


def parse_definitions(prompt_tokens: List[str]) -> Dict[str, str]:
    """Extract explicit ``M <cmd> means <op> .`` definitions from a prompt."""
    out = {}
    for i, t in enumerate(prompt_tokens):
        if t == "M":
            cmd, kw, op = prompt_tokens[i + 1:i + 4]
            assert kw == "means"
            out[cmd] = op
    return out


def answer_prompt(prompt: str, meanings: Mapping[str, str] | None = None) -> int:
    """Compute the answer to a full rendered prompt.

    The prompt ends with ``Q <program> ; <var> ->``.  If ``meanings`` is not
    given, meanings are taken from explicit definitions in the prompt.
    """
    toks = prompt.split()
    if toks and toks[0] == "<bos>":
        toks = toks[1:]
    if meanings is None:
        meanings = parse_definitions(toks)
    q = len(toks) - 1 - toks[::-1].index("Q")
    body = toks[q + 1:]
    assert body[-1] == "->", "prompt must end with '<var> ->'"
    var = body[-2]
    program = " ".join(body[:-2])
    return query(program, var, meanings)


def infer_meanings_from_demos(prompt: str) -> Dict[str, set]:
    """Return, for every command demonstrated in ``prompt``, the set of ops that
    are consistent with *all* of its demonstrations.  Used only for testing
    that demonstrations are unambiguous."""
    toks = prompt.split()
    consistent: Dict[str, set] = {}
    i = 0
    while i < len(toks):
        if toks[i] == "D":
            j = toks.index(".", i)
            demo = toks[i + 1:j]
            arrow = demo.index("->")
            result = int(demo[arrow + 1])
            var = demo[arrow - 1]
            program = " ".join(demo[:arrow - 1])
            cmds = {t for t in demo if t not in BUILTIN_OPS and t.isalpha()
                    and len(t) > 1 and t != "means"}
            assert len(cmds) == 1, demo
            cmd = cmds.pop()
            ok = {op for op in BUILTIN_OPS
                  if query(program, var, {cmd: op}) == result}
            consistent[cmd] = consistent.get(cmd, set(BUILTIN_OPS)) & ok
            i = j + 1
        else:
            i += 1
    return consistent
