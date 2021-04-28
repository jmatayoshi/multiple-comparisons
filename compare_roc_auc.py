# Basic implementation of the fast version of DeLong's test for comparing
# AUROC values.

# References:

# E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson. Comparing the areas
# under two or more correlated receiver operating characteristic curves: a
# nonparametric approach. Biometrics, pages 837 845, 1988.

# X. Sun and W. Xu. Fast implementation of DeLong s algorithm for comparing the
# areas under correlated receiver operating characteristic curves. IEEE Signal
# Processing Letters, 21(11):1389 1393, 2014

import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import norm


@wrap_non_picklable_objects    
def mid_ranks(Z):
    # Algorithm 1 from Sun and Xu (2014)
    M = len(Z)
    I = np.argsort(Z)
    W = np.r_[Z[I], Z[I][-1] + 1]
    T = np.zeros(M)
    T_Z = np.zeros(M)
    i = 0    
    while i < M:
        a = i
        j = a
        while W[j] == W[a]:
            j += 1
        b = j - 1
        for k in range(a, b + 1):
            # Add 2 to adjust for Python zero-indexing
            T[k] = (a + b + 2) / 2
        i = b + 1
    T_Z[I] = T
    
    return T_Z

@wrap_non_picklable_objects    
def fast_delong(Z, m):
    # Algorithm 2 from Sun and Xu (2014)
    k = Z.shape[1]
    n = Z.shape[0] - m    
    theta = np.zeros(k)
    V_10 = np.zeros((m, k))
    V_01 = np.zeros((n, k))
    S_10 = np.zeros((k, k))
    S_01 = np.zeros((k, k))    
    for r in range(k):
        T_Z = mid_ranks(Z[:, r])
        T_X = mid_ranks(Z[:m, r])
        T_Y = mid_ranks(Z[m:, r])
        
        V_10[:, r] = (T_Z[:m] - T_X) / n
        V_01[:, r] = 1 - (T_Z[m:] - T_Y) / m
        theta[r] = (T_Z[:m].sum() / (m*n)) - ((m + 1)/(2*n))
    S_10 = np.cov((V_10 - theta).T)
    S_01 = np.cov((V_01 - theta).T)
            
    S = S_10 / m + S_01 / n

    z_stat = (theta[0] - theta[1])/np.sqrt(S[0, 0] + S[1, 1] - 2*S[0, 1])
    p_val = 2*(1 - norm.cdf(np.abs(z_stat)))
    
    return theta, p_val, z_stat
