import numpy as np
import time
from ResRamQt import load_input, g

def optimized_g(t, obj):
    # g = ((obj.D / obj.L) ** 2) * (obj.L * t - 1 + np.exp(-obj.L * t)) + 1j * (
    #     (obj.beta * obj.D**2) / (2 * obj.L)
    # ) * (1 - np.exp(-obj.L * t))

    DL_ratio_sq = (obj.D / obj.L) ** 2
    term2_coef = 1j * (obj.beta * obj.D**2) / (2 * obj.L)

    exp_lt = np.exp(-obj.L * t)

    g_val = DL_ratio_sq * (obj.L * t - 1 + exp_lt) + term2_coef * (1 - exp_lt)
    return g_val

def test():
    obj = load_input()
    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    st = time.time()
    for _ in range(1000):
        res1 = g(thth, obj)
    old_time = time.time() - st

    st = time.time()
    for _ in range(1000):
        res2 = optimized_g(thth, obj)
    new_time = time.time() - st

    print("Match:", np.allclose(res1, res2))
    print("Old time:", old_time)
    print("New time:", new_time)

test()
