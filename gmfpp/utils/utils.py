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
    if not file == None:
        file.write("{} | {}\n".format(clock, s))
        file.flush()
