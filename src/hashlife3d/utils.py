import math


def int_log2(n):
    if n <= 0:
        raise ValueError("n must be positive")
    if math.log2(n) % 1 != 0:
        raise ValueError(f"{n} is not a power of 2")
    return int(math.log2(n))
