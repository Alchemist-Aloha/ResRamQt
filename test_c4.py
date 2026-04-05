import numpy as np
import time
from math import factorial
from ResRamQt import load_input, g, A, cross_sections

def opt_cross_sections(obj):
    time1 = time.time()
    sqrt2 = np.sqrt(2)
    obj.S = (obj.delta**2) / 2  # calculate in cross_sections()
    obj.EL = np.linspace(
        obj.E0 - obj.EL_reach, obj.E0 + obj.EL_reach, 1000
    )  # range for spectra cm^-1
    # Calculate parameters D and L based on obj attributes
    obj.D = (
        obj.gamma * (1 + 0.85 * obj.k + 0.88 * obj.k**2) / (2.355 + 1.76 * obj.k)
    )  # D parameter
    obj.L = obj.k * obj.D  # LAMBDA parameter
    obj.convEL = np.linspace(
        obj.E0 - obj.EL_reach * 0.5,
        obj.E0 + obj.EL_reach * 0.5,
        (max(len(obj.E0_range), len(obj.EL)) - min(len(obj.E0_range), len(obj.EL)) + 1),
    )

    if obj.theta == 0.0:
        H = 1.0  # np.ones(len(p.E0_range))
    else:
        H = (1 / (obj.theta * np.sqrt(2 * np.pi))) * np.exp(
            -((obj.E0_range) ** 2) / (2 * obj.theta**2)
        )

    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth - g(thth, obj)) * A(thth, obj)
    K_f = np.exp(1j * (ELEL - (obj.E0)) * thth - np.conj(g(thth, obj))) * np.conj(
        A(thth, obj)
    )

    time0 = time.time()
    ## If the order desired is 1 use the simple first order approximation ##
    if obj.order == 1:
        term2 = (((1 + obj.eta) ** 0.5 * obj.delta) / sqrt2)
        phase = obj.wg[:, np.newaxis] * thth[0, :][np.newaxis, :]
        term3 = 1 - np.exp(-1j * phase)

        q_r_diag = term2[:, np.newaxis] * term3 # shape: (len(wg), len(th))

        # calculate integration using numpy's trapezoid directly via matmul
        # K_a is (len(EL), len(th)). q_r_diag is (len(wg), len(th))
        # Trapezoidal rule for uniform grid dx=1.0: sum inner, half endpoints
        # Let's create weights for trapezoid
        weights = np.ones(len(obj.th))
        weights[0] = 0.5
        weights[-1] = 0.5

        q_r_diag_w = q_r_diag * weights

        # integ_r[l, e] = np.sum(K_a[e, t] * q_r_diag_w[l, t], axis=t)
        # We can write this as matrix multiplication!
        # K_a is shape (e, t)
        # q_r_diag_w is shape (l, t) -> q_r_diag_w.T is shape (t, l)
        # Output is shape (e, l) which we can transpose to (l, e)
        integ_r = (K_a @ q_r_diag_w.T).T

        print("Time taken for K_r calculation: ", time.time() - time0)

    elif obj.order > 1:
        pass

    time0 = time.time()
    integ_a = np.trapezoid(K_a, axis=1)
    abs_cross = (
        obj.preA * obj.convEL * np.convolve(integ_a, np.real(H), "valid") / (np.sum(H))
    )
    print("Time taken for Abs cross section calculation: ", time.time() - time0)

    time0 = time.time()
    integ_f = np.trapezoid(K_f, axis=1)
    fl_cross = (
        obj.preF * obj.convEL * np.convolve(integ_f, np.real(H), "valid") / (np.sum(H))
    )
    print("Time taken for Fl cross section calculation: ", time.time() - time0)

    time0 = time.time()

    if obj.order == 1:
        # Vectorized Raman Cross
        integ_r_sq = integ_r * np.conj(integ_r) # shape (len(wg), len(EL))

        # we need to convolve each row of integ_r_sq with H
        H_real = np.real(H)
        conv_res = np.zeros((len(obj.wg), len(obj.convEL)), dtype=complex)
        for l in range(len(obj.wg)):
            conv_res[l, :] = np.convolve(integ_r_sq[l, :], H_real, "valid")

        term3_rc = (obj.convEL[np.newaxis, :] - obj.wg[:, np.newaxis]) ** 3
        raman_cross = (obj.preR * obj.convEL * term3_rc * conv_res) / np.sum(H_real)

    elif obj.order > 1:
        pass # Handle > 1 later if needed
    print("Time taken for Raman cross section calculation: ", time.time() - time0)

    return abs_cross, fl_cross, raman_cross

def test():
    obj = load_input()

    st = time.time()
    a1, f1, r1, _, _ = cross_sections(obj)
    old_time = time.time() - st

    st = time.time()
    a2, f2, r2 = opt_cross_sections(obj)
    new_time = time.time() - st

    print("Abs match:", np.allclose(a1, a2))
    print("Fl match:", np.allclose(f1, f2))
    print("Raman match:", np.allclose(r1, r2))
    print("Old time:", old_time)
    print("New time:", new_time)

test()
