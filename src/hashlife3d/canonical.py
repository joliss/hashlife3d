from functools import wraps
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


_intern_pool = {}


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


def intern_helper(func, *args):
    key = (func, *(_to_immutable(arg) for arg in args))
    ret = _intern_pool.get(key)
    if ret is not None:
        func.cache_stats.hits += 1
        return ret
    else:
        func.cache_stats.misses += 1
        # if func.__name__ == 'generate_octree' and args[0].width() == 4:
        #     print(f'miss {func.__name__} {args[0].width()}')
        if func.cache_stats.misses % 1000 == 0:
            if func.__name__ == 'generate_octree':
                print(f'miss {func.__name__} {args[0].width()}')
            else:
                print(f'miss {func.__name__}')
                pass
        obj = func(*args)
        _intern_pool[key] = obj
        return obj


def interned(new_function):
    new_function.cache_stats = CacheStats()
    @wraps(new_function)
    def wrapper(*args):
        return intern_helper(new_function, *args)
    return wrapper


class Canonical:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @interned
    def __new__(cls, *args):
        return super().__new__(cls)

def _to_immutable(obj):
    if type(obj) is np.ndarray:
        # Trust that the items in the array are immutable
        return obj.tobytes()
    assert type(obj) not in (dict, tuple, list)
    return obj
