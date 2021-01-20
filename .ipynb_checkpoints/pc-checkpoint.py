import numpy as np

def pc_minimize(c, A, b, tol=1e-9, maxit=500):
    m, n = A.shape
    
    # Initialization
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    λ = np.linalg.lstsq(A.T, c, rcond=None)[0]
    s = c - A.T @ λ
    
    x += max(-1.5 * x.min(), 0)
    s += max(-1.5 * s.min(), 0)
    
    xts = 0.5 * x @ s
    add_x = xts / s.sum()
    add_s = xts / x.sum()
    x += add_x
    s += add_s
    
    μ = (x @ s) / n 
    nit = 0
    
    while μ > tol and nit < maxit:
        print("Iteration {}: μ = {}".format(nit, μ))
        nit += 1
        
        r_c = A.T @ λ + s - c
        r_b = A @ x - b
        
        # KKT matrix used for computing step dirs
        KKT = np.block([[np.zeros((n, n)), A.T, np.eye(n)],
                        [A, np.zeros((m, m + n))],
                        [np.diag(s), np.zeros_like(A.T), np.diag(x)]])
        Q, R = np.linalg.qr(KKT)
        
        # Predictor step
        right_aff = np.hstack([-r_c, -r_b, -x * s])
        out = np.linalg.solve(R, Q.T @ right_aff)
        Δx_aff = out[:n]
        Δλ_aff = out[n:(n + m)]
        Δs_aff = out[(n + m):]
        
        # Use predictor step output and heuristics to set up correction params
        α_pri_aff = min(1, (-x / Δx_aff)[Δx_aff < 0].min())
        α_dua_aff = min(1, (-s / Δs_aff)[Δs_aff < 0].min())
        μ_aff = ((x + α_pri_aff * Δx_aff) @ (s + α_dua_aff * Δs_aff)) / n
        σ = (μ_aff / μ)**3

        # Compute step dirs
        right = np.hstack([-r_c, -r_b, -x * s - Δx_aff * Δs_aff + σ * μ * np.ones(n)])
        out = np.linalg.solve(R, Q.T @ right)
        Δx = out[:n]
        Δλ = out[n:(n + m)]
        Δs = out[(n+m):]
        
        # Compute primal and dual step size
        nu = 0.9 ** (1 / nit)
        α_pri = min(1, nu * (-x / Δx)[Δx < 0].min())
        α_dua = min(1, nu * (-s / Δs)[Δs < 0].min())
        
        # Update
        x += α_pri * Δx
        λ += α_dua * Δλ
        s += α_dua * Δs
        μ = (x @ s) / n
        
    print("Terminated in {} iterations with μ = {}".format(nit, μ))
    return x, λ, s