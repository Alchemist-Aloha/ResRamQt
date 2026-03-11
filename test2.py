from line_profiler import LineProfiler
from ResRamQt import A, g, load_input

def run_test():
    obj = load_input()
    import numpy as np
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)
    A(thth, obj)
    g(thth, obj)

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(A)
    lp.add_function(g)
    lp_wrapper = lp(run_test)
    lp_wrapper()
    lp.print_stats()
