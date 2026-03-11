import numpy as np
from line_profiler import LineProfiler
from ResRamQt import load_input, g, A

def run_test():
    obj = load_input()
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth - g(thth, obj)) * A(thth, obj)
    K_f = np.exp(1j * (ELEL - (obj.E0)) * thth - np.conj(g(thth, obj))) * np.conj(A(thth, obj))

    dx = obj.th[1] - obj.th[0] if len(obj.th) > 1 else 1.0
    w = np.ones(len(obj.th))
    w[0] = 0.5
    w[-1] = 0.5

    # Check if these operations can be optimized
    integ_a_old = np.trapezoid(K_a, axis=1)
    integ_a_new = dx * (K_a @ w)

    print(np.allclose(integ_a_old, integ_a_new))

if __name__ == '__main__':
    run_test()
