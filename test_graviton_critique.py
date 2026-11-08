import numpy as np
import matplotlib.pyplot as plt

def build_exact_toda_params(n, ell, graviton_amp=0.0):
    N = n + 1
    W0 = np.zeros((N, N))
    for k in range(N):
        if k < n: W0[k+1, k] = (n - k) / n
        if k > 0: W0[k-1, k] = k / n
    W0 = (1 - ell) * W0 + ell * np.eye(N)
    
    # === INJECTION DU GRAVITON (Onde de spin-2) ===
    if graviton_amp > 0:
        h = np.zeros((N, N))
        for i in range(N - 1):
            val = graviton_amp * np.sin(2 * np.pi * 2 * i / N)
            h[i, i+1] = val
            h[i+1, i] = val
        W0 = W0 + h
    # ===============================================
    
    eigvals, eigvecs = np.linalg.eig(W0)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi0 = np.real(eigvecs[:, idx])
    if np.sum(pi0) < 0: pi0 = -pi0
    pi0 /= np.sum(pi0)

    D_sqrt = np.diag(np.sqrt(pi0))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(pi0))
    J = D_inv_sqrt @ W0 @ D_sqrt
    J = 0.5 * (J + J.T)
    return J

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
    n = 31
    ell = 0.45
    dt = 0.01
    max_t = 35.0
    m = 15 
    
    # On teste 3 amplitudes de graviton
    amplitudes = [0.0, 0.05, 0.10]
    couleurs = ['blue', 'orange', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for amp, color in zip(amplitudes, couleurs):
        J = build_exact_toda_params(n, ell, graviton_amp=amp)
        t = 0.0
        times, min_eigs = [], []
        
        while t < max_t:
            J_block = J[0:m, 0:m]
            eigvals = np.linalg.eigvalsh(J_block)
            min_eig = np.min(np.abs(eigvals))
            
            times.append(t)
            min_eigs.append(min_eig)
            
            J = rk4_step(J, dt)
            t += dt
            
        plt.plot(times, min_eigs, label=f"Amplitude graviton = {amp}", color=color, linewidth=2)

    plt.yscale('log')
    plt.title(f"Impact du Graviton sur la plus petite valeur propre du bloc J_1:{m}")
    plt.xlabel("Temps t")
    plt.ylabel("|lambda_min| (échelle log)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("impact_graviton.png")
    print("Graphique sauvegardé : impact_graviton.png")
    plt.show()
