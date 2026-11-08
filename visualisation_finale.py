import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ==========================================
# 1. FONCTIONS DE BASE
# ==========================================
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

def inject_perturbation(J0, h):
    J = J0.copy()
    i = np.arange(len(h))
    J[i, i + 1] += h
    J[i + 1, i] += h
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

def exact_derivative(J, v):
    dJdt = toda_rhs(J)
    return float(v @ dJdt @ v)

# ==========================================
# 2. COLLECTE DES DONNÉES ET ERREUR
# ==========================================
def collect_data(J_start, dt=0.001, max_t=10.0):
    J = J_start.copy()
    t = 0.0
    ts, cross, dex, dnum = [], [], [], []
    
    w, v = np.linalg.eigh(J)
    idx = np.argmin(np.abs(w))
    lam_prev = w[idx]
    
    while t <= max_t:
        w, v = np.linalg.eigh(J)
        idx = np.argmin(np.abs(w))
        lam = w[idx]
        
        ts.append(t)
        cross.append(lam)
        dex.append(exact_derivative(J, v[:, idx]))
        
        if len(cross) > 1:
            dnum.append((lam - lam_prev) / dt)
        else:
            dnum.append(0.0)
            
        lam_prev = lam
        J = rk4_step(J, dt)
        t += dt
        
    return np.array(ts), np.array(cross), np.array(dex), np.array(dnum)

if __name__ == "__main__":
    J0 = build_base_J()
    np.random.seed(42)
    h = np.random.uniform(-0.1, 0.1, 31) 
    J_pert = inject_perturbation(J0, h)

    print("Simulation en cours...")
    t0, c0, de0, dn0 = collect_data(J0)
    t1, c1, de1, dn1 = collect_data(J_pert)

    # Calcul de l'erreur
    err0 = np.abs(de0 - dn0)
    err1 = np.abs(de1 - dn1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # --- Plot 1 : Valeurs propres ---
    axes[0].plot(t0, c0, label='Témoin', color='teal', linewidth=2)
    axes[0].plot(t1, c1, label='Perturbé', color='orange', linewidth=2)
    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].set_ylabel('Crossing Eigenvalue')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2 : Dérivées (Exacte vs Numérique) ---
    axes[1].plot(t0, de0, label='Témoin - Exacte', color='teal', linewidth=2)
    axes[1].plot(t0, dn0, label='Témoin - Numérique', color='teal', linestyle=':')
    axes[1].plot(t1, de1, label='Perturbé - Exacte', color='orange', linewidth=2)
    axes[1].plot(t1, dn1, label='Perturbé - Numérique', color='orange', linestyle=':')
    axes[1].set_ylabel('d lambda / dt')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3 : Erreur ---
    axes[2].plot(t0, err0, label='Erreur Témoin', color='teal')
    axes[2].plot(t1, err1, label='Erreur Perturbé', color='orange')
    axes[2].set_ylabel('Erreur absolue |exact - num|')
    axes[2].set_xlabel('t')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("preuve_derniere_erreur.png", dpi=300)
    print("Graphique sauvegardé : preuve_derniere_erreur.png")
    plt.show()
