import numpy as np
from line_profiler import LineProfiler
from ResRamQt import load_input

def run_test():
    obj = load_input()
    sqrt2 = np.sqrt(2)
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)
    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth)

    # K_r calculation
    q_r = np.ones((len(obj.wg), len(obj.wg), len(obj.th)), dtype=complex)
    K_r = np.zeros((len(obj.wg), len(obj.EL), len(obj.th)), dtype=complex)

    # Optimize
    term1 = 1.0 / np.sqrt(1) # since q[idxl] is always 1 for the diagonal in obj.Q which is identity matrix
    term2 = ((1 + obj.eta) ** 0.5 * obj.delta) / sqrt2
    term3 = 1 - np.exp(-1j * np.outer(obj.wg, obj.th))

    for idxq, q in enumerate(obj.Q):
        for idxl, l in enumerate(q):
            if l > 0:
                q_r[idxq, idxl, :] = term1 * term2[idxl] * term3[idxl, :]

        K_r[idxq, :, :] = K_a * np.prod(q_r, axis=1)[idxq]

    integ_r = np.trapezoid(K_r, axis=2)

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(run_test)
    lp_wrapper = lp(run_test)
    lp_wrapper()
    lp.print_stats()
