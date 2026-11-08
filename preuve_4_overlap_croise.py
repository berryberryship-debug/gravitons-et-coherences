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

def get_eigenvectors(J, m=15): 
    w, v = np.linalg.eigh(J[:m, :m])
    idx = np.argsort(w)
    return w[idx], v[:, idx]

if __name__ == "__main__":
    J0 = build_base_J()
    p = 0.05
    mode = 7.75
    A_crit = 0.275
    
    amplitudes = np.linspace(0.26, 0.29, 50)
    overlaps_croises = []
    gaps = []
    
    print("Calcul de l'overlap croisé (Branche 1 vs Branche 2)...")
    for A in amplitudes:
        J_pert = inject_graviton(J0, A, mode, p)
        w, v = get_eigenvectors(J_pert)
        
        # Les deux branches
        v1 = v[:, 0]
        v2 = v[:, 1]
        
        # Overlap (produit scalaire absolu)
        overlap = np.abs(np.dot(v1, v2))
        overlaps_croises.append(overlap)
        
        # Gap
        gaps.append(np.abs(w[1] - w[0]))
        
    # Tracé avec deux axes Y
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Amplitude (A)')
    ax1.set_ylabel('Overlap entre Branche 1 et 2', color='green')
    ax1.plot(amplitudes, overlaps_croises, 'o-', color='green', label='Overlap croisé')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.axvline(A_crit, color='red', linestyle='--', label='Point critique A=0.275')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Gap (log)', color='blue')
    ax2.plot(amplitudes, gaps, 's--', color='blue', alpha=0.5, label='Gap')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()
    plt.title("Preuve 4 : Overlap croisé des états (Transition Topologique)")
    plt.grid(True, alpha=0.3)
    plt.savefig("preuve_4_overlap_croise.png")
    print("Graphique sauvegardé : preuve_4_overlap_croise.png")
    plt.show()
