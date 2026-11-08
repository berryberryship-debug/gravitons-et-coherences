import numpy as np

def build_curie_weiss(n, beta):
    N = n + 1
    W = np.zeros((N, N))
    for k in range(N):
        if k < n:
            f_up = (2*k - n + 1) / n
            p_up = ((n - k) / n) * (1 / (1 + np.exp(-2 * beta * f_up)))
            W[k+1, k] = p_up
        if k > 0:
            f_down = (2*k - n - 1) / n
            p_down = (k / n) * (1 - 1 / (1 + np.exp(-2 * beta * f_down)))
            W[k-1, k] = p_down
        W[k, k] = 1.0 - np.sum(W[:, k])
    return W

def build_ehrenfest(n):
    N = n + 1
    W = np.zeros((N, N))
    for k in range(N):
        if k < n:
            W[k+1, k] = (n - k) / n
        if k > 0:
            W[k-1, k] = k / n
    return W

def build_lazy_ehrenfest(n):
    W = build_ehrenfest(n)
    return 0.5 * (np.eye(n + 1) + W)

def run_test(W0, name, N=32):
    n = N - 1
    print(f"\n--- Test : {name} (N={N}) ---")
    
    # Jauge de signe
    eigvals, eigvecs = np.linalg.eig(W0)
    idx = np.argmin(np.abs(eigvals - 1))
    pi0 = np.real(eigvecs[:, idx])
    if np.sum(pi0) < 0: pi0 = -pi0
    pi0 /= np.sum(pi0)
    
    D_sqrt = np.diag(np.sqrt(pi0))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(pi0))
    J0 = D_inv_sqrt @ W0 @ D_sqrt
    J0 = (J0 + J0.T) / 2
    
    spectre0 = np.sort(np.linalg.eigvalsh(J0))
    print(f"Lambda min initial : {np.min(spectre0):.5f}")
    
    def step_rk4(J, dt):
        def B(J_mat): return np.tril(J_mat, -1) - np.triu(J_mat, 1)
        def f(J_mat):
            b = B(J_mat)
            return b @ J_mat - J_mat @ b
        k1 = f(J)
        k2 = f(J + 0.5 * dt * k1)
        k3 = f(J + 0.5 * dt * k2)
        k4 = f(J + dt * k3)
        return J + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    J_t = J0.copy()
    dt = 0.1
    tc = None
    
    for step in range(11):
        t = step * dt
        eigvals_t, eigvecs_t = np.linalg.eigh(J_t)
        idx_max = np.argmax(eigvals_t)
        u1 = eigvecs_t[:, idx_max]
        
        if np.sum(u1) < 0: u1 = -u1 # Correction Jauge
        
        V_diag = np.diag(u1)
        V_inv = np.diag(1.0 / u1)
        W_t = V_diag @ J_t @ V_inv
        
        positivity = np.min(W_t)
        stochasticity = np.max(np.abs(np.sum(W_t, axis=0) - 1))
        spectral_error = np.max(np.abs(np.sort(eigvals_t) - spectre0))
        
        if positivity < -1e-12 and tc is None: # Correction test min W
            tc = t
            print(f"-> RUPTURE à t={tc:.2f} : Min(W) < 0 ({positivity:.2e})")
            break
            
        print(f"t={t:.1f} | Min(W)={positivity: .4e} | Err.Stoch={stochasticity:.1e} | Err.Spec={spectral_error:.1e}")
        
        J_t = step_rk4(J_t, dt)

    if tc is None:
        print("[CONCLUSION] Pas de rupture détectée sur l'intervalle.")

if __name__ == "__main__":
    N = 32
    run_test(build_curie_weiss(N-1, 1.0), "Curie-Weiss (beta=1)", N)
    run_test(build_ehrenfest(N-1), "Ehrenfest", N)
    run_test(build_lazy_ehrenfest(N-1), "Ehrenfest Paresseux", N)
