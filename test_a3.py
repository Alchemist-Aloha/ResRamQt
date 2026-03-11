import numpy as np
from ResRamQt import load_input, A
import time

def optimized_A3(t, obj):
    a = (1 + obj.eta) * obj.S
    b = obj.eta * obj.S
    # sum_a = np.sum(a)
    # sum_b = np.sum(b)
    # sum_ab = sum_a + sum_b
    sum_ab = np.sum(a + b)

    # t can be scalar, 1D array, or 2D sparse meshgrid (1, len(th)).
    # To handle all robustly, we compute a*exp(-i w t) using tensordot

    # Actually wg is 1D array. t is arbitrary shape.
    # wg*t is computed as np.multiply.outer(wg, t) or wg[:, np.newaxis...] * t
    # For a general solution:
    # tensordot(a, exp(-1j * np.multiply.outer(wg, t)), axes=(0, 0))

    # Let's see if this is fast:
    wg_t = np.multiply.outer(obj.wg, t) # shape: (len(wg), *t.shape)

    exp_minus = np.exp(-1j * wg_t)
    exp_plus = np.conj(exp_minus) # exp(1j * x) is conj(exp(-1j * x)) since wg_t is real!

    # sum over wg (axis 0 of wg_t)
    sum_K = sum_ab - np.tensordot(a, exp_minus, axes=1) - np.tensordot(b, exp_plus, axes=1)
    return obj.M**2 * np.exp(-sum_K)

def test():
    obj = load_input()
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    st = time.time()
    for _ in range(100):
        res1 = A(thth, obj)
    old_time = time.time() - st

    st = time.time()
    for _ in range(100):
        res2 = optimized_A3(thth, obj)
    new_time = time.time() - st

    print("Match:", np.allclose(res1, res2))
    print("Old time:", old_time)
    print("New time:", new_time)

test()
