
import inspect
import os
def debug_print(*args):
    frame = inspect.currentframe()
    caller = inspect.getframeinfo(frame.f_back)
    print(f"{os.path.basename(caller.filename)}:{caller.lineno} -", *args)