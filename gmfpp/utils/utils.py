import torch
import numpy as np

def get_clock_time():
    from time import gmtime, strftime
    result = strftime("%H:%M:%S", gmtime())
    return result

def get_datetime():
    from time import gmtime, strftime
    result = strftime("%Y-%m-%d - %H-%M-%S", gmtime())
    return result
    
def cprint(s: str, file=None):    
    clock = get_clock_time()
    print("{} | {}".format(clock, s))

    if file:
        file.write("{} | {}".format(clock, s))
        file.flush()
    
def create_logfile(filepath: str):
    return open(filepath, "w")

def constant_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
