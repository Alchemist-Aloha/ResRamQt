import numpy as np
from ResRamQt import load_input, g, A

obj = load_input()
thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

K_a = np.exp(1j * (ELEL - (obj.E0)) * thth - g(thth, obj)) * A(thth, obj)
K_f = np.exp(1j * (ELEL - (obj.E0)) * thth - np.conj(g(thth, obj))) * np.conj(A(thth, obj))

integ_a_old = np.trapezoid(K_a, obj.th, axis=1)
integ_a_new = np.trapezoid(K_a, dx=obj.th[1]-obj.th[0], axis=1)

print("K_a diff:", np.max(np.abs(integ_a_old - integ_a_new)))

w = np.ones(len(obj.th))
w[0] = 0.5
w[-1] = 0.5
dx = obj.th[1] - obj.th[0]

integ_a_dot = (K_a @ w) * dx

print("K_a dot diff:", np.max(np.abs(integ_a_old - integ_a_dot)))
