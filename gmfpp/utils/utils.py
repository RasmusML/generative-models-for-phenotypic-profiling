def get_clock_time():
    from time import gmtime, strftime
    result = strftime("%H:%M:%S", gmtime())
    return result
    
def cprint(s: str):
    clock = get_clock_time()
    print("{} | {}".format(clock, s))