import inspect

import numpy as np


def print_params():
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    args = frame.f_locals

    max_param_length = max(len(param) for param in args.keys())

    print(f"IMAGE PROCESSED WITH METHOD: {func_name}")
    for param, value in args.items():
        if not isinstance(value, np.ndarray):
            print(f"\t{param:<{max_param_length}}\t{value}")