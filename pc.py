import numpy as np

def pc_minimize(c, A, b, tol=1e-9, maxit=500):
    m, n = A.shape
    
    # Initialization
    x = np.linalg.lstsq(A, b, rcond=None)[0]     # == A.T @ np.linalg.inv(A @ A.T) @ b
    λ = np.linalg.lstsq(A.T, c, rcond=None)[0]   # == np.linalg.inv(A @ A.T) @ A @ c
    s = c - A.T @ λ
    
    x += max(-1.5 * x.min(), 0)
    s += max(-1.5 * s.min(), 0)
    
    xts = 0.5 * x @ s
    add_x = xts / s.sum()
    add_s = xts / x.sum()
    x += add_x
    s += add_s
    
    mu = (x @ s) / n 
    nit = 0
    
    while mu > tol and nit < maxit:
        print("Iteration {}: mu = {}".format(nit, mu))
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
        delta_x_aff = out[:n]
        delta_λ_aff = out[n:(n + m)]
        delta_s_aff = out[(n + m):]
        
        # Use predictor step output and heuristics to set up correction params
        alpha_pri_aff = min(1, (-x / delta_x_aff)[delta_x_aff < 0].min())
        alpha_dua_aff = min(1, (-s / delta_s_aff)[delta_s_aff < 0].min())
        mu_aff = ((x + alpha_pri_aff * delta_x_aff) @ (s + alpha_dua_aff * delta_s_aff)) / n
        sigma = (mu_aff / mu)**3

        # Compute step dirs
        right = np.hstack([-r_c, -r_b, -x * s - delta_x_aff * delta_s_aff + sigma * mu * np.ones(n)])
        out = np.linalg.solve(R, Q.T @ right)
        delta_x = out[:n]
        delta_λ = out[n:(n + m)]
        delta_s = out[(n+m):]
        
        # Compute primal and dual step size
        nu = 0.9 ** (1 / nit)
        alpha_pri = min(1, nu * (-x / delta_x)[delta_x < 0].min())
        alpha_dua = min(1, nu * (-s / delta_s)[delta_s < 0].min())
        
        # Update
        x += alpha_pri * delta_x
        λ += alpha_dua * delta_λ
        s += alpha_dua * delta_s
        mu = (x @ s) / n
        
    print("Terminated in {} iterations with mu = {}".format(nit, mu))
    return x, λ, s