from typing import Iterator


_do_show_progress = True
def set_do_show_progress(value):
    global _do_show_progress
    _do_show_progress = value


def unwrap_with_progress_bar(iterator):
    assert isinstance(iterator, Iterator)
    try:
        n = 0
        while True:
            progress = next(iterator)
            n += 1
            if n % 1000 == 0 and _do_show_progress:
                print(f'\rProgress: {progress}', end="", flush=True)
    except StopIteration as e:
        print(flush=True)
        return e.value
