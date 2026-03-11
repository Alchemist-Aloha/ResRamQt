import numpy as np
from line_profiler import LineProfiler
from ResRamQt import load_input

def run_test():
    obj = load_input()
    sqrt2 = np.sqrt(2)
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)
    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth)

    # optimize K_r logic
    term1 = 1.0 / np.sqrt(1)
    term2 = ((1 + obj.eta) ** 0.5 * obj.delta) / sqrt2
    term3 = 1 - np.exp(-1j * np.outer(obj.wg, obj.th))

    q_r_diag = term1 * term2[:, np.newaxis] * term3 # shape: (len(wg), len(th))

    # Using dot product for integration (Riemann sum) or trapezoidal explicitly:
    # np.trapezoid(y, x, axis) -> sum of (y_i + y_i+1) * dx / 2
    # But since dx is uniform: dx = obj.th[1] - obj.th[0]
    # sum along th: K_a @ q_r_diag.T

    # K_a is shape (len(EL), len(th)), q_r_diag is shape (len(wg), len(th))
    # We want integral over `th`.
    # Using Riemann sum for comparison: K_a @ q_r_diag.T gives shape (len(EL), len(wg))

    dx = obj.th[1] - obj.th[0] if len(obj.th) > 1 else 1.0
    # Actually trapezoid rule for uniform dx over axis:
    # y[0]*dx/2 + y[-1]*dx/2 + sum(y[1:-1])*dx
    # = (sum(y) - y[0]/2 - y[-1]/2) * dx

    # We want this applied to K_a(e,t) * q_r_diag(l,t)

    # Let's adjust q_r_diag weights
    w = np.ones(len(obj.th))
    w[0] = 0.5
    w[-1] = 0.5

    q_r_weighted = q_r_diag * w
    integ_r_new = dx * (K_a @ q_r_weighted.T).T # Output shape: (len(wg), len(EL))

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(run_test)
    lp_wrapper = lp(run_test)
    lp_wrapper()
    lp.print_stats()
