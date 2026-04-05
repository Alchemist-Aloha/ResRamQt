import numpy as np
from ResRamQt import load_input, A
import time

def optimized_A2(t, obj):
    if isinstance(t, np.ndarray):
        # We only need A(t). t is usually thth which has shape (1, len(th))
        # The equation for K[l] depends only on t!
        # A depends on sum(K) over l.
        # sum_K = sum_l { (1+eta)*S*(1 - exp(-1j*wg*t)) + eta*S*(1 - exp(1j*wg*t)) }
        # Let a_l = (1+eta_l)*S_l
        # Let b_l = eta_l*S_l
        # K_l(t) = a_l(1 - exp(-1j*wg_l*t)) + b_l(1 - exp(1j*wg_l*t))
        # K_l(t) = a_l + b_l - a_l*exp(-1j*wg_l*t) - b_l*exp(1j*wg_l*t)
        # sum_K(t) = sum(a+b) - sum(a_l*exp(-1j*wg_l*t)) - sum(b_l*exp(1j*wg_l*t))

        a = (1 + obj.eta) * obj.S
        b = obj.eta * obj.S

        # t is (1, len(th))
        wg_t = obj.wg[:, np.newaxis] * t[0, :][np.newaxis, :] # shape: (len(wg), len(th))

        term1 = a @ np.exp(-1j * wg_t) # dot product? No, wait.
        # sum_l a_l * exp(-1j * wg_l * t) is just dot product of a and exp matrix!

        exp_minus = np.exp(-1j * wg_t)
        exp_plus = np.exp(1j * wg_t)

        sum_K = np.sum(a + b) - (a @ exp_minus) - (b @ exp_plus)

        # A_val = obj.M**2 * exp(-sum_K)
        # But A shape needs to be broadcastable to t
        A_val = obj.M**2 * np.exp(-sum_K)
        # sum_K is 1D (len(th)), but t was (1, len(th))
        # so A_val is (len(th),)
        return A_val[np.newaxis, :]
    else:
        return A(t, obj)

def test():
    obj = load_input()
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    st = time.time()
    for _ in range(100):
        res1 = A(thth, obj)
    old_time = time.time() - st

    st = time.time()
    for _ in range(100):
        res2 = optimized_A2(thth, obj)
    new_time = time.time() - st

    print("Match:", np.allclose(res1, res2))
    print("Old time:", old_time)
    print("New time:", new_time)

test()
