import numpy as np

def build_base_J(n: int = 31, ell: float = 0.45) -> np.ndarray:
    N = n + 1
    W0 = np.zeros((N, N))
    k_sub = np.arange(n)
    W0[k_sub + 1, k_sub] = (n - k_sub) / n
    k_sup = np.arange(1, N)
    W0[k_sup - 1, k_sup] = k_sup / n
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

def inject_graviton(J_start: np.ndarray, amp: float, mode: float, p: float) -> np.ndarray:
    J = J_start.copy()
    n = J.shape[0] - 1
    idx = np.arange(n)
    val = amp * np.sin(2 * np.pi * mode * idx / n)
    J[idx, idx + 1] = (J[idx, idx + 1] + val) * (1.0 - p)
    J[idx + 1, idx] = (J[idx + 1, idx] + val) * (1.0 - p)
    return J

def toda_rhs(J: np.ndarray) -> np.ndarray:
    B = np.tril(J, -1) - np.triu(J, 1)
    return B @ J - J @ B

def rk4_step(J: np.ndarray, dt: float) -> np.ndarray:
    k1 = toda_rhs(J)
    k2 = toda_rhs(J + 0.5 * dt * k1)
    k3 = toda_rhs(J + 0.5 * dt * k2)
    k4 = toda_rhs(J + dt * k3)
    return J + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def get_eigenvector(J_start: np.ndarray, amp: float, t_target: float, mode: float, p: float, dt: float = 0.001) -> np.ndarray:
    J = inject_graviton(J_start, amp, mode, p)
    num_steps = int(t_target / dt)
    for _ in range(num_steps):
        J = rk4_step(J, dt)
    t_current = num_steps * dt
    if t_target > t_current:
        J = rk4_step(J, t_target - t_current)
    w, v = np.linalg.eigh(J[:15, :15])
    idx = np.argmin(w)
    return v[:, idx]

if __name__ == "__main__":
    J0 = build_base_J()
    mode = 7.75
    p = 0.05
    A0 = 0.274885
    t0 = 0.467794
    R = 0.0005
    N_points = 50
    thetas = np.linspace(0, 2*np.pi, N_points)
    
    print(f"Calcul de la phase de Berry (V2 - Jauge Fixée)...")
    
    vectors = []
    for theta in thetas:
        A = A0 + R * np.cos(theta)
        t = t0 + R * np.sin(theta)
        v = get_eigenvector(J0, A, t, mode, p)
        vectors.append(v)
        
    # Lissage de la jauge
    for i in range(len(vectors) - 1):
        overlap = np.dot(vectors[i], vectors[i+1])
        if overlap < 0:
            vectors[i+1] = -vectors[i+1]
            
    # Test topologique à la fermeture
    overlap_fermeture = np.dot(vectors[-1], vectors[0])
    
    if overlap_fermeture < 0:
        berry_phase = np.pi
    else:
        berry_phase = 0.0
    
    print("\n>>> RÉSULTAT (V2) <<<")
    print(f"Phase de Berry accumulée = {berry_phase:.4f} radians")
    print(f"Soit environ {berry_phase / np.pi:.4f} * pi")
    
    if np.isclose(berry_phase, np.pi):
        print("CONCLUSION : La phase est exacte à pi. C'est un VRAI point diabolique (singularité topologique) !")
    else:
        print("CONCLUSION : La phase est à 0. C'est un évitement de croisement trivial.")
