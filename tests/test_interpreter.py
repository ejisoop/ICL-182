import pytest
from src.data import interpreter as I
from src.data.tokenizer import MODULUS


@pytest.mark.parametrize("op,old,arg,exp", [
    ("set", 2, 3, 3), ("add", 2, 3, 5), ("sub", 2, 3, (2 - 3) % MODULUS),
    ("mul", 5, 4, 20 % MODULUS), ("add", 12, 12, 24 % MODULUS), ("set", 7, 0, 0),
])
def test_apply_op(op, old, arg, exp):
    assert I.apply_op(op, old, arg) == exp


def test_builtin_program():
    env = I.execute("a = 2 ; b = 5 ; add a b ; mul b 2 ; sub a 1 ; set b a")
    assert env == {"a": 6, "b": 6}


def test_made_up_command_meanings():
    prog = "a = 2 ; dax a 3"
    assert I.query(prog, "a", {"dax": "set"}) == 3
    assert I.query(prog, "a", {"dax": "add"}) == 5
    assert I.query(prog, "a", {"dax": "sub"}) == (2 - 3) % MODULUS
    assert I.query(prog, "a", {"dax": "mul"}) == 6


def test_variable_argument_and_self_reference():
    assert I.query("a = 4 ; dax a a", "a", {"dax": "mul"}) == 16 % MODULUS


def test_errors():
    with pytest.raises(I.InterpreterError):
        I.execute("a = 1 ; dax a 2")  # no meaning
    with pytest.raises(I.InterpreterError):
        I.execute("add z 1")  # undefined var
    with pytest.raises(I.InterpreterError):
        I.query("a = 1", "b")


def test_answer_prompt_definition_and_explicit():
    p = "<bos> M dax means add . Q a = 2 ; dax a 3 ; a ->"
    assert I.answer_prompt(p) == 5
    p2 = "<bos> D b = 2 ; dax b 4 ; b -> 4 . Q a = 2 ; dax a 3 ; a ->"
    assert I.answer_prompt(p2, {"dax": "set"}) == 3
    assert I.infer_meanings_from_demos(p2) == {"dax": {"set"}}
