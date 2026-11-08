import numpy as np
import matplotlib.pyplot as plt

def build_base_J(n=31, ell=0.45):
    N = n + 1
    W0 = np.zeros((N, N))
    for k in range(N):
        if k < n: W0[k+1, k] = (n - k) / n
        if k > 0: W0[k-1, k] = k / n
    W0 = (1 - ell) * W0 + ell * np.eye(N)
    eigvals, eigvecs = np.linalg.eig(W0)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi0 = np.real(eigvecs[:, idx])
    if np.sum(pi0) < 0: pi0 = -pi0
    pi0 /= np.sum(pi0)
    D_sqrt = np.diag(np.sqrt(pi0))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(pi0))
    J0 = D_inv_sqrt @ W0 @ D_sqrt
    return 0.5 * (J0 + J0.T)

def inject_graviton(J0, amp, mode, p, n=31):
    J = J0.copy()
    for i in range(n):
        val = amp * np.sin(2 * np.pi * mode * i / n)
        J[i, i + 1] += val
        J[i + 1, i] += val
        J[i, i + 1] *= (1.0 - p)
        J[i + 1, i] *= (1.0 - p)
    return J

def toda_rhs(J):
    B = np.tril(J, -1) - np.triu(J, 1)
    return B @ J - J @ B

def rk4_step(J, dt):
    k1 = toda_rhs(J)
    k2 = toda_rhs(J + 0.5*dt*k1)
    k3 = toda_rhs(J + 0.5*dt*k2)
    k4 = toda_rhs(J + dt*k3)
    return J + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

if __name__ == "__main__":
    J0 = build_base_J()
    p = 0.05
    mode = 7.75
    A_crit = 0.275
    
    J_pert = inject_graviton(J0, A_crit, mode, p)
    
    dt = 0.005
    max_t = 5.0
    t = 0.0
    
    times = []
    gaps = []
    
    J = J_pert.copy()
    while t <= max_t:
        # On prend le bloc 15x15
        w = np.linalg.eigvalsh(J[:15, :15])
        # On trie et on prend les deux plus petites
        w = np.sort(w)
        gap = np.abs(w[1] - w[0])
        
        times.append(t)
        gaps.append(gap)
        
        J = rk4_step(J, dt)
        t += dt

    plt.figure(figsize=(10, 5))
    plt.plot(times, gaps, color='blue', linewidth=2)
    plt.yscale('log')
    plt.title(f"Évolution temporelle du Gap à A={A_crit} (Mode Diabolique)")
    plt.xlabel("Temps t")
    plt.ylabel("Gap |\lambda_1 - \lambda_2| (log)")
    plt.grid(True, alpha=0.3)
    plt.show()
