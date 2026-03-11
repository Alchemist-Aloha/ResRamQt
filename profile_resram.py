import cProfile
import pstats
import io
import sys
from ResRamQt import load_input, cross_sections

def profile():
    obj = load_input()
    pr = cProfile.Profile()
    pr.enable()

    # Run the expensive calculation
    abs_cross, fl_cross, raman_cross, boltz_state, boltz_coef = cross_sections(obj)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == '__main__':
    profile()
