from functools import wraps
from collections.abc import Iterable
from dataclasses import dataclass
import time

import numpy as np


_intern_pool = {}
do_log = True


@dataclass
class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.last_log = time.perf_counter_ns()

    def log_miss(self, name, n):
        now = time.perf_counter_ns()
        microseconds_per_calculation = (now - self.last_log) / 1000 / n
        self.last_log = now
        print(f'{self.misses:10n} calls to {name} ({microseconds_per_calculation:.2f} us avg)', flush=True)


def unwrap_progress_iterable(iterator):
    try:
        while True:
            next(iterator)
    except StopIteration as e:
        return e.value


def intern_helper(func, *args, do_log):
    assert do_log == False, 'do_log not implemented'
    key = (func, *(_to_immutable(arg) for arg in args))
    ret = _intern_pool.get(key)
    if ret is not None:
        if do_log:
            func.cache_stats.hits += 1
        return ret
    else:
        if do_log:
            func.cache_stats.misses += 1
        obj = func(*args)
        _intern_pool[key] = obj
        return obj


def intern_helper_with_progress(func, *args, do_log, progress_range):
    key = (func, *(_to_immutable(arg) for arg in args))
    ret = _intern_pool.get(key)
    if ret is not None:
        if do_log:
            func.cache_stats.hits += 1
        # yield progress_range[0]
        return ret
    else:
        # print(f'Cache miss, range={progress_range}')
        iterator = func(*args)
        assert isinstance(iterator, Iterable)
        try:
            while True:
                progress = next(iterator)
                assert not len(progress_range[1]) or len(progress_range[0]) + len(progress) == len(progress_range[1])
                yield progress_range[0] + progress
        except StopIteration as e:
            ret = e.value
        _intern_pool[key] = ret
        yield progress_range[1]
        if do_log:
            func.cache_stats.misses += 1
            if func.cache_stats.misses % 1000 == 0:
                func.cache_stats.log_miss(func.__name__, 1000)
        return ret


def interned(_function=None, *, do_log=False):
    def decorator(new_function):
        new_function.cache_stats = CacheStats()
        @wraps(new_function)
        def wrapper(*args):
            return intern_helper(new_function, *args, do_log=do_log)
        return wrapper
    if _function is None:
        return decorator
    else:
        return decorator(_function)


def interned_with_progress(_function=None, *, do_log=False):
    if do_log:
        from .progress import set_do_show_progress
        set_do_show_progress(False)
    def decorator(new_function):
        new_function.cache_stats = CacheStats()
        @wraps(new_function)
        def wrapper(*args, progress_range=((), ())):
            nonlocal do_log
            return (yield from intern_helper_with_progress(new_function, *args, progress_range=progress_range, do_log=do_log))
        return wrapper
    if _function is None:
        return decorator
    else:
        return decorator(_function)


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
