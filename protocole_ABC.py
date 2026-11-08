import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FONCTIONS DE BASE (Construction & Flot)
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

# ==========================================
# 2. OBSERVABLES (Déterminant et Valeurs Propres)
# ==========================================
def get_D_m(J, m=15):
    """Calcule le déterminant du bloc m x m via la récurrence des mineurs."""
    b = np.diag(J).copy()
    a = np.diag(J, k=1).copy()
    D = np.zeros(len(b))
    D[0] = b[0]
    if len(b) > 1:
        D[1] = b[1]*D[0] - a[0]**2
        for k in range(2, len(b)):
            D[k] = b[k]*D[k-1] - a[k-1]**2 * D[k-2]
    return D[m-1]

def get_min_eig(J, m=15):
    """Calcule la plus petite valeur propre en module du bloc m x m."""
    w = np.linalg.eigvalsh(J[:m, :m])
    return np.min(np.abs(w))

def get_inertie(J, m=15):
    """Calcule le nombre de valeurs propres négatives du bloc m x m."""
    w = np.linalg.eigvalsh(J[:m, :m])
    return int((w < 0).sum())

# ==========================================
# 3. RECHERCHE DU TEMPS CRITIQUE (BISSECTION)
# ==========================================
def trouver_tc_bissection(J_start, dt_initial=0.01, max_t=20.0):
    """Avance le temps jusqu'au changement de signe de D_15, puis affine."""
    J = J_start.copy()
    t = 0.0
    D_prec = get_D_m(J)
    
    # Phase 1 : Avancer à grands pas pour trouver l'intervalle
    while t < max_t:
        J_next = rk4_step(J, dt_initial)
        D_curr = get_D_m(J_next)
        
        if D_prec * D_curr < 0:
            # Changement de signe détecté entre t et t+dt_initial
            t_debut = t
            t_fin = t + dt_initial
            break
            
        J = J_next
        D_prec = D_curr
        t += dt_initial
    else:
        return None, None, None, None # Pas de rupture trouvée

    # Phase 2 : Bissection fine pour trouver le zéro exact
    J_debut = J.copy() # Sauvegarde de l'état à t_debut
    dt_fin = 0.0001
    
    while (t_fin - t_debut) > dt_fin:
        t_milieu = (t_debut + t_fin) / 2.0
        steps = int((t_milieu - t_debut) / dt_fin)
        
        J_temp = J_debut.copy()
        for _ in range(steps):
            J_temp = rk4_step(J_temp, dt_fin)
            
        D_milieu = get_D_m(J_temp)
        
        if D_prec * D_milieu <= 0:
            t_fin = t_milieu
        else:
            t_debut = t_milieu
            D_prec = D_milieu
            
    # On a trouvé t_c
    tc = (t_debut + t_fin) / 2.0
    J_tc = J_debut.copy()
    steps_tc = int(tc / dt_fin)
    for _ in range(steps_tc):
        J_tc = rk4_step(J_tc, dt_fin)
        
    return tc, get_D_m(J_tc), get_min_eig(J_tc), get_inertie(J_tc)

# ==========================================
# 4. EXÉCUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":
    J0 = build_base_J()
    
    # Création d'une perturbation (on prend l'optimisée si tu l'as, sinon aléatoire)
    np.random.seed(42)
    h = np.random.uniform(-0.1, 0.1, 31) 
    J_pert = inject_perturbation(J0, h)
    
    print("--- ANALYSE DE LA TRANSITION TOPOLOGIQUE (D_15) ---")
    
    # Test du témoin (sans perturbation)
    print("\n[1] Témoin (sans perturbation) :")
    tc0, D0, eig0, nu0 = trouver_tc_bissection(J0)
    if tc0:
        print(f"  -> Rupture à t_c = {tc0:.4f}")
        print(f"  -> D_15(t_c) = {D0:.2e}")
        print(f"  -> Lambda_min(t_c) = {eig0:.2e}")
        print(f"  -> Inertie à t_c = {nu0}")
    else:
        print("  -> Pas de rupture.")

    # Test avec la perturbation
    print("\n[2] Avec perturbation (vecteur aléatoire) :")
    tc1, D1, eig1, nu1 = trouver_tc_bissection(J_pert)
    if tc1:
        print(f"  -> Rupture à t_c = {tc1:.4f}")
        print(f"  -> D_15(t_c) = {D1:.2e}")
        print(f"  -> Lambda_min(t_c) = {eig1:.2e}")
        print(f"  -> Inertie à t_c = {nu1}")
    else:
        print("  -> Pas de rupture.")
