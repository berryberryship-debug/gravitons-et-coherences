import numpy as np
import matplotlib.pyplot as plt
import time

# --- Paramètres ---
N, M, ELL = 32, 15, 0.45
C_MAX, MARGE = 0.20, 1e-2
DT, CAP = 0.02, 4.0

def base_J(n=N - 1, ell=ELL):
    W = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        if k < n: W[k + 1, k] = (n - k) / n
        if k > 0: W[k - 1, k] = k / n
    W = (1 - ell) * W + ell * np.eye(n + 1)
    ev, vec = np.linalg.eig(W)
    p = np.real(vec[:, np.argmin(abs(ev - 1))])
    p = p * np.sign(p.sum()); p /= p.sum()
    J = np.diag(1 / np.sqrt(p)) @ W @ np.diag(np.sqrt(p))
    return 0.5 * (J + J.T)

def inject(J0, h):
    J = J0.copy()
    i = np.arange(len(h))
    J[i, i + 1] += h
    J[i + 1, i] += h
    return J

def toda(J):
    B = np.tril(J, -1) - np.triu(J, 1)
    return B @ J - J @ B

def rk4(J, dt):
    k1 = toda(J); k2 = toda(J + .5 * dt * k1)
    k3 = toda(J + .5 * dt * k2); k4 = toda(J + dt * k3)
    return J + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def inertie(J):
    w = np.linalg.eigvalsh(J[:M, :M])
    return int((w < 0).sum()), float(np.abs(w).min())

def faisable(J, n_ref):
    n, mn = inertie(J)
    return n == n_ref and mn >= MARGE

def T1(J, dt=DT, cap=CAP):
    n0 = inertie(J)[0]
    t = 0.0
    while t < cap:
        J = rk4(J, dt); t += dt
        if inertie(J)[0] != n0:
            return t
    return cap + inertie(J)[1]

def evolution(J0, x0, n_ref, budget, rng):
    best = x0.copy(); fbest = T1(inject(J0, best)); sig = 0.05; ev = 1
    while ev < budget:
        cand = np.clip(best + sig * rng.normal(size=best.size), -C_MAX, C_MAX)
        Jc = inject(J0, cand)
        if not faisable(Jc, n_ref):
            sig *= 0.98; continue
        f = T1(Jc); ev += 1
        if f <= fbest:
            best, fbest = cand, f; sig = min(0.15, sig * 1.2)
        else:
            sig = max(0.005, sig * 0.9)
    return best, fbest

if __name__ == "__main__":
    print("Lancement de l'optimisation... (patientez quelques secondes)")
    J0 = base_J()
    n_ref, marge0 = inertie(J0)
    t_tem = T1(J0)
    
    # Optimisation
    rng = np.random.default_rng(0)
    x0 = np.zeros(31)
    h_best, f_best = evolution(J0, x0, n_ref, budget=400, rng=rng)
    
    print(f"Témoin T1 = {t_tem:.3f}")
    print(f"Optimisé T1 = {f_best:.3f}")
    
    # --- Génération de l'onde parfaite ---
    h_opt = h_best
    positions = np.arange(len(h_opt))
    
    plt.figure(figsize=(12, 5))
    
    # 1. Forme de l'onde
    plt.subplot(1, 2, 1)
    plt.stem(positions, h_opt, basefmt="k-")
    plt.title("Profil de la perturbation optimale \(h_i\)")
    plt.xlabel("Site \(i\)")
    plt.ylabel("Amplitude")
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # 2. Spectre de Fourier
    plt.subplot(1, 2, 2)
    spectre = np.abs(np.fft.rfft(h_opt))
    freqs = np.fft.rfftfreq(len(h_opt))
    plt.plot(freqs[1:], spectre[1:], 'o-')
    plt.title("Spectre de Fourier (Modes spatiaux)")
    plt.xlabel("Fréquence spatiale")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("arme_absolue.png")
    print("Graphique sauvegardé : arme_absolue.png")
    plt.show()
