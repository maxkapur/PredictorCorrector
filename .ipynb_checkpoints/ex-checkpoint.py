import numpy as np
from scipy.optimize import linprog
from pc import pc_minimize

# Data structure that yields feasible problems.
n, m = 500, 200
A = np.block([np.random.rand(m, n), np.eye(m)])
b = np.random.randint(50, 100, m)
c = np.hstack([np.random.randint(-200, -100, n), np.zeros(m)])

out_pc = pc_minimize(c, A, b)
out_lp = linprog(c, A_eq = A, b_eq = b)
print("\nScipy linprog terminated in {} iterations".format(out_lp.nit))

print("\nTotal absolute deviation:")
print(np.abs(out_pc[0] - out_lp.x).sum())
