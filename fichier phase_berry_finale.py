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

def toda_rhs(J):
    B = np.tril(J, -1) - np.triu(J, 1)
    return B @ J - J @ B

def rk4_step(J, dt):
    k1 = toda_rhs(J)
    k2 = toda_rhs(J + 0.5*dt*k1)
    k3 = toda_rhs(J + 0.5*dt*k2)
    k4 = toda_rhs(J + dt*k3)
    return J + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def get_eigenvector(J_start, amp, t_target, mode, p, dt=0.001):
    """Calcule le vecteur propre de la plus petite valeur propre au temps t_target."""
    J = inject_graviton(J0, amp, mode, p)
    t = 0.0
    while t < t_target - 1e-9:
        step = min(dt, t_target - t)
        J = rk4_step(J, step)
        t += step
        
    w, v = np.linalg.eigh(J[:15, :15])
    idx = np.argmin(w)
    return v[:, idx]

if __name__ == "__main__":
    global J0
    J0 = build_base_J()
    mode = 7.75
    p = 0.05
    
    # Point critique trouvé par Nelder-Mead
    A0 = 0.274885
    t0 = 0.467794
    
    # Paramètres du cercle entourant le point critique
    R = 0.0005 # Rayon du cercle (très petit)
    N_points = 50 # Nombre de points sur le cercle
    
    thetas = np.linspace(0, 2*np.pi, N_points)
    
    print(f"Calcul de la phase de Berry autour de (A={A0}, t={t0})...")
    
    # On calcule les vecteurs propres le long du cercle
    vectors = []
    for theta in thetas:
        A = A0 + R * np.cos(theta)
        t = t0 + R * np.sin(theta)
        
        # On s'assure que les paramètres restent dans les limites physiques
        if A < 0.26 or A > 0.29 or t < 0.4 or t > 0.55:
            continue 
            
        v = get_eigenvector(J0, A, t, mode, p)
        vectors.append(v)
        
    # Calcul de la phase de Berry (holonomie)
    berry_phase = 0.0
    for i in range(len(vectors)-1):
        overlap = np.dot(vectors[i], vectors[i+1])
        berry_phase += np.angle(overlap)
        
    # Fermeture de la boucle (dernier point avec le premier)
    overlap_fermeture = np.dot(vectors[-1], vectors[0])
    berry_phase += np.angle(overlap_fermeture)
    
    print(f"\n>>> RÉSULTAT <<<")
    print(f"Phase de Berry accumulée = {berry_phase:.4f} radians")
    print(f"Soit environ {berry_phase / np.pi:.4f} * pi")
    
    if np.abs(np.abs(berry_phase) - np.pi) < 0.5:
        print("CONCLUSION : La phase est proche de pi. C'est un VRAI point diabolique (singularité topologique) !")
    else:
        print("CONCLUSION : La phase est proche de 0. C'est un évitement de croisement.")
