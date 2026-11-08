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
    mode = 7.75
    p = 0.05
    
    # Grille allégée (20x20)
    amps = np.linspace(0.270, 0.280, 20)
    times = np.linspace(0.40, 0.55, 20)
    
    gap_matrix = np.zeros((len(times), len(amps)))
    
    print("Génération du cône (version rapide)...")
    
    for j, A in enumerate(amps):
        J_pert = inject_graviton(J0, A, mode, p)
        t = 0.0
        dt_fin = 0.005
        
        for i, t_target in enumerate(times):
            # Intégration jusqu'au temps cible
            while t < t_target - 1e-9:
                J_pert = rk4_step(J_pert, dt_fin)
                t += dt_fin
            
            # Gap
            w = np.sort(np.linalg.eigvalsh(J_pert[:15, :15]))
            gap_matrix[i, j] = np.abs(w[1] - w[0])
            
    # Minimum
    min_idx = np.unravel_index(np.argmin(gap_matrix), gap_matrix.shape)
    best_t = times[min_idx[0]]
    best_A = amps[min_idx[1]]
    best_gap = gap_matrix[min_idx]
    
    print(f"\nMinimum trouvé : Gap = {best_gap:.4e} (A={best_A:.4f}, t={best_t:.4f})")
    
    # Tracé
    plt.figure(figsize=(8, 5))
    plt.imshow(gap_matrix, aspect='auto', cmap='viridis', 
               extent=[amps[0], amps[-1], times[-1], times[0]],
               norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Gap spectral (log)')
    plt.plot(best_A, best_t, 'r*', markersize=15, label='Minimum')
    plt.xlabel("Amplitude A")
    plt.ylabel("Temps t")
    plt.title("Cône de Dégénérescence (Version Rapide)")
    plt.legend()
    plt.savefig("cone_rapide.png")
    plt.show()
