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

def inject_perturbation_2D(J0, amp, mode, n=31):
    J = J0.copy()
    for i in range(n):
        # On fait varier l'amplitude ET la fréquence spatiale (mode)
        val = amp * np.sin(2 * np.pi * mode * i / n)
        J[i, i + 1] += val
        J[i + 1, i] += val
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

def get_min_gap_2D(J_start, dt=0.01, max_t=10.0):
    J = J_start.copy()
    t = 0.0
    w, v = np.linalg.eigh(J)
    # On prend les 2 valeurs propres les plus proches de 0
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
    
    # Grille de scan
    amplitudes = np.linspace(0.0, 0.3, 25)
    modes = np.arange(1, 10) # Modes de 1 à 9
    
    # Matrice pour stocker les écarts
    gap_matrix = np.zeros((len(modes), len(amplitudes)))
    
    print("Scan 2D en cours... (patientez)")
    for i, mode in enumerate(modes):
        for j, amp in enumerate(amplitudes):
            J_pert = inject_perturbation_2D(J0, amp, mode)
            gap = get_min_gap_2D(J_pert)
            gap_matrix[i, j] = gap
        print(f"Mode {mode} terminé.")
            
    # Tracé de la carte thermique
    plt.figure(figsize=(10, 6))
    # On utilise une échelle log pour mieux voir les petits écarts
    plt.imshow(gap_matrix, aspect='auto', cmap='viridis', 
               extent=[amplitudes[0], amplitudes[-1], modes[-1], modes[0]],
               norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Écart minimal (log)')
    plt.xlabel("Amplitude (Graviton)")
    plt.ylabel("Mode spatial (Fréquence)")
    plt.title("Carte thermique de l'écart spectral (Recherche de singularité)")
    plt.show()
