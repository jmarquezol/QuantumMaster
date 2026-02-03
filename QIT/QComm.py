import numpy as np
from scipy.optimize import brentq

def binary_entropy(x):
    if x <=0 or x >= 1:
        return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def binary_entropy2(x):
    if x <=0 or x >= 1:
        return 0
    return (1+x)*0.5 * np.log2(1+x) + (1 - x)*0.5 * np.log2(1 - x)

def total_entropy(p):
    # Eigenvalues (lambda)
    p0 = (1 + 3*p) / 4
    p1 = (1 - p) / 4
    # S(rho) = -sum(lambda log lambda)
    # We have one p0 and three p1
    return -p0 * np.log2(p0) - 3 * p1 * np.log2(p1)

def objective(p):
    # We want f(p) = S(rho_AB) - 1 = 0
    #return total_entropy(p) - 1
    return binary_entropy2(p) - binary_entropy(0.5*(1+np.sqrt(2*p**2-1)))

# Find root in range [0, 1]
p_crit = brentq(objective, 0.71, 0.999)
print(f"Critical depolarization p: {p_crit:.4f}")
