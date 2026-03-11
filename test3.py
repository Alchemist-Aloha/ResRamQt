from line_profiler import LineProfiler
from ResRamQt import load_input, cross_sections

def run_test():
    obj = load_input()
    for _ in range(10):
        cross_sections(obj)

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(cross_sections)
    lp_wrapper = lp(run_test)
    lp_wrapper()
    lp.print_stats()
