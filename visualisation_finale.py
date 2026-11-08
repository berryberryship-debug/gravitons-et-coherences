import numpy as np
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

def get_D_m(J, m=15):
    b = np.diag(J).copy()
    a = np.diag(J, k=1).copy()
    D = np.zeros(len(b))
    D[0] = b[0]
    if len(b) > 1:
        D[1] = b[1]*D[0] - a[0]**2
        for k in range(2, len(b)):
            D[k] = b[k]*D[k-1] - a[k-1]**2 * D[k-2]
    return D[m-1]

def get_inertie(J, m=15):
    w = np.linalg.eigvalsh(J[:m, :m])
    return int((w < 0).sum())

# ==========================================
# 2. SIMULATION
# ==========================================
def simuler(J_start, dt=0.01, max_t=12.0):
    J = J_start.copy()
    t = 0.0
    times, D_vals, inerties = [], [], []
    
    while t <= max_t:
        times.append(t)
        D_vals.append(get_D_m(J))
        inerties.append(get_inertie(J))
        J = rk4_step(J, dt)
        t += dt
        
    return np.array(times), np.array(D_vals), np.array(inerties)

if __name__ == "__main__":
    J0 = build_base_J()
    np.random.seed(42)
    h = np.random.uniform(-0.1, 0.1, 31) 
    J_pert = inject_perturbation(J0, h)

    t_star_temoin = 1.768858
    t_star_pert = 9.793832

    print("Simulation en cours...")
    t0, D0, I0 = simuler(J0)
    t1, D1, I1 = simuler(J_pert)

    # Création de la figure avec 2 panneaux explicites
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Panneau 1 : Déterminant D_15 ---
    ax1.plot(t0, D0, color='teal', label='Témoin', linewidth=2)
    ax1.plot(t1, D1, color='orange', label='Perturbé', linewidth=2)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax1.axvline(t_star_temoin, color='teal', linestyle='--', alpha=0.7)
    ax1.axvline(t_star_pert, color='orange', linestyle='--', alpha=0.7)
    
    ax1.set_title("Déterminant \(D_{15}(t)\) (Passage par zéro)")
    ax1.set_xlabel("Temps t")
    ax1.set_ylabel("\(D_{15}\)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panneau 2 : Inertie (Spectral Flow) ---
    ax2.step(t0, I0, color='teal', label='Témoin', where='post', linewidth=2.5)
    ax2.step(t1, I1, color='orange', label='Perturbé', where='post', linewidth=2.5)
    
    ax2.axvline(t_star_temoin, color='teal', linestyle='--', alpha=0.7)
    ax2.axvline(t_star_pert, color='orange', linestyle='--', alpha=0.7)
    
    ax2.set_title("Spectral Flow : Nombre de valeurs propres négatives")
    ax2.set_xlabel("Temps t")
    ax2.set_ylabel("Inertie")
    ax2.set_yticks([1, 2, 3, 4])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("preuve_spectral_flow_complete.png", dpi=300)
    print("Graphique sauvegardé : preuve_spectral_flow_complete.png")
    
    # Force l'affichage
    plt.show()
