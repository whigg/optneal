from functools import wraps
import time


def execute_time(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print('{}: {}[s]'.format(func.__name__, elapsed_time))
        return elapsed_time
    return wrapper