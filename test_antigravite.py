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

def inject_graviton_and_pressure(J0, amp, mode, pressure, n=31):
    J = J0.copy()
    for i in range(n):
        val = amp * np.sin(2 * np.pi * mode * i / n)
        J[i, i + 1] += val
        J[i + 1, i] += val
    for i in range(n + 1):
        J[i, i] -= pressure
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

def get_min_gap(J_start, dt=0.01, max_t=10.0):
    J = J_start.copy()
    t = 0.0
    w, v = np.linalg.eigh(J)
    indices = np.argsort(np.abs(w))[:2]
    branches_w = [[], []]
    branches_v = [v[:, indices[0]], v[:, indices[1]]]
    
    while t <= max_t:
        w, v = np.linalg.eigh(J)
        for b in range(2):
            v_prev = branches_v[b]
            overlaps = np.abs(v_prev @ v)
            idx_new = int(np.argmax(overlaps))
            branches_w[b].append(w[idx_new])
            branches_v[b] = v[:, idx_new]
        J = rk4_step(J, dt)
        t += dt
        
    b1 = np.array(branches_w[0])
    b2 = np.array(branches_w[1])
    return np.min(np.abs(b1 - b2))

if __name__ == "__main__":
    J0 = build_base_J()
    mode_opt = 3
    amp_opt = 0.10
    
    # ZOOM FIN autour de 0.29
    pressures = np.linspace(0.25, 0.35, 30) 
    gaps = []
    
    print("Zoom fin autour de la résonance (x=0.29)...")
    min_gap = float('inf')
    best_p = 0
    
    for p in pressures:
        J_pert = inject_graviton_and_pressure(J0, amp_opt, mode_opt, p)
        gap = get_min_gap(J_pert)
        gaps.append(gap)
        
        if gap < min_gap:
            min_gap = gap
            best_p = p
            
        print(f"Champ externe = {p:.4f} | Écart = {gap:.4e}")
        
    print(f"\n>>> Minimum absolu trouvé : {min_gap:.4e} à x = {best_p:.4f}")
        
    plt.figure(figsize=(10, 5))
    plt.plot(pressures, gaps, 'o-', color='darkred', linewidth=2)
    plt.title("Zoom fin : Effet de l'Anti-Gravité sur l'Évitement de Croisement")
    plt.xlabel("Force du Potentiel (Anti-Gravité)")
    plt.ylabel("Écart minimal (Masse)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
