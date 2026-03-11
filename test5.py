from line_profiler import LineProfiler
from ResRamQt import load_input, cross_sections
import math

def run_test():
    obj = load_input()
    import numpy as np

    # K_r calculation
    sqrt2 = np.sqrt(2)
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)
    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth) # simplified
    q_r = np.ones((len(obj.wg), len(obj.wg), len(obj.th)), dtype=complex)
    K_r = np.zeros((len(obj.wg), len(obj.EL), len(obj.th)), dtype=complex)

    for idxq, q in enumerate(obj.Q, start=0):
        for idxl, l in enumerate(q, start=0):
            if q[idxl] > 0:
                q_r[idxq, idxl, :] = (
                    (1.0 / math.factorial(q[idxl])) ** (0.5)
                    * (((1 + obj.eta[idxl]) ** (0.5) * obj.delta[idxl]) / sqrt2)
                    ** (q[idxl])
                    * (1 - np.exp(-1j * obj.wg[idxl] * thth)) ** (q[idxl])
                )
            elif q[idxl] < 0:
                q_r[idxq, idxl, :] = (
                    (1.0 / math.factorial(np.abs(q[idxl]))) ** (0.5)
                    * (((obj.eta[l]) ** (0.5) * obj.delta[l]) / sqrt2) ** (-q[idxl])
                    * (1 - np.exp(1j * obj.wg[idxl] * thth)) ** (-q[idxl])
                )
        K_r[idxq, :, :] = K_a * (np.prod(q_r, axis=1)[idxq])

    integ_r = np.zeros((len(obj.wg), len(obj.EL)), dtype=complex)
    for l in range(len(obj.wg)):
        integ_r[l, :] = np.trapezoid(K_r[l, :, :], axis=1)

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(run_test)
    lp_wrapper = lp(run_test)
    lp_wrapper()
    lp.print_stats()
