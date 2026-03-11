import numpy as np
from line_profiler import LineProfiler
from ResRamQt import load_input

def run_test():
    obj = load_input()
    sqrt2 = np.sqrt(2)
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)
    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth)

    # optimize K_r logic
    q_r_diag = np.ones((len(obj.wg), len(obj.th)), dtype=complex)

    term1 = 1.0 / np.sqrt(1)
    term2 = ((1 + obj.eta) ** 0.5 * obj.delta) / sqrt2
    term3 = 1 - np.exp(-1j * np.outer(obj.wg, obj.th))

    q_r_diag = term1 * term2[:, np.newaxis] * term3

    # Instead of full K_r calculation and integration:
    # K_r[l, :, :] = K_a * q_r_diag[l, :]
    # integ_r = np.trapezoid(K_r, axis=2)
    #
    # We can perform integration directly on K_a * q_r_diag
    # Actually K_a is (len(EL), len(th)), q_r_diag is (len(wg), len(th))
    # we want to integrate over `th` (axis=1 of K_a)

    # Using np.trapz(y, x, axis) -> np.trapezoid(y, dx=... or x=..., axis)
    # If using uniform spacing, we can just sum or use trapezoid rule.
    # We want integral of K_a[e, t] * q_r_diag[l, t] dt

    dt = obj.th[1] - obj.th[0] if len(obj.th) > 1 else 1.0

    # Just broadcasting
    # shape: (len(wg), len(EL), len(th))
    K_r = K_a[np.newaxis, :, :] * q_r_diag[:, np.newaxis, :]
    integ_r = np.trapezoid(K_r, axis=2)

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(run_test)
    lp_wrapper = lp(run_test)
    lp_wrapper()
    lp.print_stats()
