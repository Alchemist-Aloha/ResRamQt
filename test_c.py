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
        # q_r[idxq, idxl, :] depends on q[idxl]
        # In fact, obj.Q is an identity matrix, so q[idxl] = 1 if idxq == idxl else 0
        # If q[idxl] = 1 (l > 0 branch logic):
        # q_r[l, l, :] = (1.0 / factorial(1))**0.5 * (((1 + eta)**0.5 * delta) / sqrt2)**1 * (1 - exp(-1j * wg * thth))**1
        # If q[idxl] = 0, q_r is supposed to be 1 since ones initialization.

        # product of q_r across axis 1 (idxl)
        # np.prod(q_r, axis=1)[idxq] -> This product over idxl is just q_r[idxq, idxq, :] because all other elements for that idxq are 1!
        # So we only need to compute the diagonal where idxq == idxl and q[idxl] == 1.

        term1 = 1.0  # (1.0 / factorial(1))**0.5
        term2 = (((1 + obj.eta) ** 0.5 * obj.delta) / sqrt2)
        # phase: exp(-1j * wg * thth)
        phase = obj.wg[:, np.newaxis] * thth[0, :][np.newaxis, :]
        term3 = 1 - np.exp(-1j * phase)

        q_r_diag = term1 * term2[:, np.newaxis] * term3

        # K_r shape (len(wg), len(EL), len(th))
        # K_r[idxq, :, :] = K_a * q_r_diag[idxq, :]
        K_r = K_a[np.newaxis, :, :] * q_r_diag[:, np.newaxis, :]
        print("Time taken for K_r calculation: ", time.time() - time0)

    elif obj.order > 1:
        # this path logic hasn't changed.
        tpp, tmm, ELEL = np.meshgrid(obj.tp, obj.tm, obj.EL, sparse=True)
        K_r = np.exp(
            1j * (ELEL - obj.E0) * sqrt2 * tmm
            - g((tpp + tmm), obj) / (sqrt2)
            - np.conj(g((tpp - tmm) / (sqrt2), obj))
        )
        integ_r1 = np.zeros((len(obj.tm), len(obj.EL)), dtype=complex)
        for idxtm, tm in enumerate(obj.tm, start=0):
            integ_r1[idxtm, :] = np.trapezoid(
                K_r[(np.abs(len(obj.tm) // 2 - idxtm)) :, idxtm, :], axis=0
            )

        integ = np.trapezoid(integ_r1, axis=0)

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
    raman_cross = np.zeros((len(obj.wg), len(obj.convEL)), dtype=complex)

    if obj.order == 1:
        integ_r = np.trapezoid(K_r, axis=2)

        # Vectorized Raman Cross
        integ_r_sq = integ_r * np.conj(integ_r) # shape (len(wg), len(EL))

        # we need to convolve each row of integ_r_sq with H
        conv_res = np.zeros((len(obj.wg), len(obj.convEL)), dtype=complex)
        for l in range(len(obj.wg)):
            conv_res[l, :] = np.convolve(integ_r_sq[l, :], np.real(H), "valid")

        term3_rc = (obj.convEL[np.newaxis, :] - obj.wg[:, np.newaxis]) ** 3
        raman_cross = (obj.preR * obj.convEL * term3_rc * conv_res) / np.sum(H)

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
