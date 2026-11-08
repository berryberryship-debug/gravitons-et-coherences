import numpy as np
import matplotlib.pyplot as plt

def build_exact_toda_params(n, ell):
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
    m = 15 # Taille du bloc (indices 0 à 14)
    
    J = build_exact_toda_params(n, ell)
    t = 0.0
    
    times = []
    min_eigenvalues = []
    
    while t < max_t:
        # On extrait le bloc J_{1:m} (les m premiers sites)
        J_block = J[0:m, 0:m]
        
        # On calcule ses valeurs propres
        eigvals = np.linalg.eigvalsh(J_block)
        
        # On garde la valeur propre la plus proche de zéro (en valeur absolue)
        min_eig = np.min(np.abs(eigvals))
        
        times.append(t)
        min_eigenvalues.append(min_eig)
        
        J = rk4_step(J, dt)
        t += dt
        
    plt.figure(figsize=(8, 5))
    plt.plot(times, min_eigenvalues, color='blue', linewidth=2)
    plt.yscale('log') # Échelle logarithmique pour voir les petites valeurs
    plt.title(f"Plus petite valeur propre (en valeur absolue) du bloc J_1:{m}")
    plt.xlabel("Temps t")
    plt.ylabel("|lambda_min| (échelle log)")
    plt.grid(True, which="both", ls="--")
    plt.savefig("audit_valeurs_propres.png")
    print("Graphique sauvegardé : audit_valeurs_propres.png")
    plt.show()
