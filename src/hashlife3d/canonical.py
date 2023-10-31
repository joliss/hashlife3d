import numpy as np


class Canonical:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    _intern_pool = {}

    def __new__(cls, *args, **kwargs):
        key = cls._key(*args, **kwargs)
        if key in cls._intern_pool:
            return cls._intern_pool[key]
        else:
            obj = super().__new__(cls)
            cls._intern_pool[key] = obj
            return obj

    @classmethod
    def _key(cls, *args, **kwargs):
        frozen_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = arg.tolist()
            if isinstance(arg, list):
                arg = _list_to_tuple_recursive(arg)
            frozen_args.append(arg)
        return (cls, tuple(frozen_args), frozenset(kwargs.items()))


def _list_to_tuple_recursive(lst):
    return tuple(_list_to_tuple_recursive(e) if isinstance(e, list) else e for e in lst)
