class Canonical:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    _intern_pool = {}

    def __new__(cls, *args, **kwargs):
        key = (cls, args, frozenset(kwargs.items()))
        if key in cls._intern_pool:
            return cls._intern_pool[key]
        else:
            obj = super().__new__(cls)
            cls._intern_pool[key] = obj
            return obj
