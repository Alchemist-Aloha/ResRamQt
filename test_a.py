import numpy as np
from ResRamQt import load_input, A
import time

def optimized_A(t, obj):
    if isinstance(t, np.ndarray):
        # K shape: (len(wg), len(t))
        # K[l, :] = (1 + eta) * S * (1 - exp(-1j * wg * t)) + eta * S * (1 - exp(1j * wg * t))

        # t can be 2D array: thth shape (len(EL), len(th))
        # wg shape (len(wg),)

        term1 = (1 + obj.eta) * obj.S
        term2 = obj.eta * obj.S

        # If t is thth (1000, 175) and wg is (26,)
        # we can compute K in a vectorized way:
        # we need exp(-1j * wg[:, np.newaxis, np.newaxis] * t[np.newaxis, :, :])

        phase = obj.wg[:, np.newaxis, np.newaxis] * t[np.newaxis, :, :]
        exp_minus = np.exp(-1j * phase)
        exp_plus = np.exp(1j * phase)

        K = term1[:, np.newaxis, np.newaxis] * (1 - exp_minus) + term2[:, np.newaxis, np.newaxis] * (1 - exp_plus)

        A_val = obj.M**2 * np.exp(-np.sum(K, axis=0))
        return A_val
    else:
        K = np.zeros((len(obj.wg), 1), dtype=complex)
        # Calculate the K matrix
        for l in np.arange(len(obj.wg)):
            K[l, :] = (1 + obj.eta[l]) * obj.S[l] * (
                1 - np.exp(-1j * obj.wg[l] * t)
            ) + obj.eta[l] * obj.S[l] * (1 - np.exp(1j * obj.wg[l] * t))
        A_val = obj.M**2 * np.exp(-np.sum(K, axis=0))
        return A_val

def test():
    obj = load_input()
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    st = time.time()
    res1 = A(thth, obj)
    old_time = time.time() - st

    st = time.time()
    res2 = optimized_A(thth, obj)
    new_time = time.time() - st

    print("Match:", np.allclose(res1, res2))
    print("Old time:", old_time)
    print("New time:", new_time)

test()
