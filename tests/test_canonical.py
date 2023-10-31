from hashlife3d.canonical import Canonical

class MyClass(Canonical):
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_canonical():
    assert MyClass(1, 2) is MyClass(1, 2)
