import numpy as np
import matplotlib.pyplot as plt

def build_base_J(n, ell):
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

def toda_rhs_matriciel(J):
    B = np.tril(J, -1) - np.triu(J, 1)
    return B @ J - J @ B

def rk4_step(J, dt):
    k1 = toda_rhs_matriciel(J)
    k2 = toda_rhs_matriciel(J + 0.5 * dt * k1)
    k3 = toda_rhs_matriciel(J + 0.5 * dt * k2)
    k4 = toda_rhs_matriciel(J + dt * k3)
    return J + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == "__main__":
    n = 31; ell = 0.45; dt = 0.01; max_t = 40.0; m = 15
    J0 = build_base_J(n, ell)
    
    # 1. Trouver le vecteur propre du bloc initial
    J_block = J0[0:m, 0:m]
    evals, evecs = np.linalg.eigh(J_block)
    
    # On prend le vecteur propre de la plus petite valeur propre (en valeur absolue)
    idx_min = np.argmin(np.abs(evals))
    v = evecs[:, idx_min]
    
    # 2. Construire la perturbation optimale : h_i = v_i * v_{i+1}
    # (On normalise pour avoir une amplitude max de 0.20)
    h = v[:-1] * v[1:]
    h = h / np.max(np.abs(h)) * 0.20
    
    # 3. Injecter dans la matrice complète
    J_opt = J0.copy()
    for i in range(len(h)):
        J_opt[i, i+1] += h[i]
        J_opt[i+1, i] += h[i]
        
    # --- Comparaison avec Mode 2 ---
    J_mode2 = J0.copy()
    for i in range(n):
        val = 0.20 * np.sin(4 * np.pi * i / n)
        J_mode2[i, i+1] += val
        J_mode2[i+1, i] += val

    # Simulation des deux
    def run_sim(J_start):
        J = J_start.copy()
        t, min_eigs = 0.0, []
        times = []
        while t < max_t:
            ev = np.linalg.eigvalsh(J[0:m, 0:m])
            times.append(t)
            min_eigs.append(np.min(np.abs(ev)))
            J = rk4_step(J, dt)
            t += dt
        return times, min_eigs

    t1, min_eig_opt = run_sim(J_opt)
    t2, min_eig_mode2 = run_sim(J_mode2)

    plt.figure(figsize=(10, 6))
    plt.plot(t1, min_eig_opt, color='red', linewidth=2, label="Perturbation Adjointe (basée sur v)")
    plt.plot(t2, min_eig_mode2, color='blue', linewidth=2, label="Mode 2 Pur")
    plt.yscale('log')
    plt.title("L'Arme Absolue : Perturbation alignée sur le mode propre")
    plt.xlabel("Temps t")
    plt.ylabel("|lambda_min| (log)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
