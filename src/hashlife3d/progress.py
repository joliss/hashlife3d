from typing import Iterator

import psutil
from pympler import asizeof

from .canonical import _intern_pool


_do_show_progress = True
def set_do_show_progress(value):
    global _do_show_progress
    _do_show_progress = value

_process = psutil.Process()

def unwrap_with_progress_bar(iterator):
    assert isinstance(iterator, Iterator)
    try:
        n = 0
        while True:
            progress = next(iterator)
            n += 1
            if n % 1000 == 0 and _do_show_progress:
                progress_stats = ' '.join(f'{n:2n}' for n in progress)
                memory = _process.memory_info().rss
                memory_stats = f', {len(_intern_pool):12n} objects, {int(memory) // 1024 // 1024:7n} MB allocated'
                # memory_stats += f', {int(asizeof.asizeof(_intern_pool) / len(_intern_pool))} accurate bytes'
                # memory_stats += f', {int(asizeof.asizeof(list(_intern_pool.keys())) / len(_intern_pool))} key bytes'
                # memory_stats += f', {int(asizeof.asizeof(list(_intern_pool.values())) / len(_intern_pool))} value bytes'
                print(f'\rProgress: {progress_stats}{memory_stats}', end="", flush=True)
    except StopIteration as e:
        print(flush=True)
        return e.value
