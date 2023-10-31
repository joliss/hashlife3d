from hashlife3d.state import State

def test_state():
    assert str(State.DEAD) == '_'
    assert State.from_str('X') == State.ALIVE
