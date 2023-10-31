from functools import wraps
from collections.abc import Iterable

import numpy as np


_intern_pool = {}


def intern_helper(func, *args, **kwargs):
    key = _to_tuple_recursive((func, args, kwargs))
    if key in _intern_pool:
        return _intern_pool[key]
    else:
        obj = func(*args, **kwargs)
        _intern_pool[key] = obj
        return obj


def interned(new_function):
    @wraps(new_function)
    def wrapper(*args, **kwargs):
        return intern_helper(new_function, *args, **kwargs)
    return wrapper


class Canonical:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @interned
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


def _to_tuple_recursive(obj):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    elif isinstance(obj, dict):
        obj = obj.items()
    if isinstance(obj, Iterable):
        return tuple(_to_tuple_recursive(e) for e in obj)
    else:
        return obj
