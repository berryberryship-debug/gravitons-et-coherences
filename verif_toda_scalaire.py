import numpy as np

def verifier_equations_toda(N=6):
    # Créer une matrice de Jacobi aléatoire (tridiagonale symétrique)
    np.random.seed(42)
    b = np.random.rand(N)      # Diagonale
    a = np.random.rand(N-1)    # Hors-diagonale
    
    J = np.diag(b) + np.diag(a, k=1) + np.diag(a, k=-1)
    
    # 1. Calcul du crochet de Lax matriciel
    B = np.tril(J, -1) - np.triu(J, 1)
    dJ_matriciel = B @ J - J @ B
    
    # 2. Calcul des équations scalaires de l'IA
    db_scalaire = np.zeros(N)
    da_scalaire = np.zeros(N-1)
    
    for k in range(N):
        a_k_moins_1_carre = a[k-1]**2 if k > 0 else 0
        a_k_carre = a[k]**2 if k < N-1 else 0
        db_scalaire[k] = 2 * (a_k_moins_1_carre - a_k_carre)
        
    for k in range(N-1):
        da_scalaire[k] = a[k] * (b[k+1] - b[k])
        
    # 3. Extraction du résultat matriciel pour comparaison
    db_matriciel = np.diag(dJ_matriciel)
    da_matriciel = np.diag(dJ_matriciel, k=1)
    
    # 4. Comparaison
    print("--- Vérification des équations de Toda ---")
    print("Erreur sur la diagonale (db) :", np.max(np.abs(db_scalaire - db_matriciel)))
    print("Erreur sur les hors-diagonales (da) :", np.max(np.abs(da_scalaire - da_matriciel)))
    
    if np.allclose(db_scalaire, db_matriciel) and np.allclose(da_scalaire, da_matriciel):
        print("[SUCCÈS] Le crochet de Lax matriciel est EXACTEMENT équivalent aux équations scalaires.")
    else:
        print("[ÉCHEC] Il y a une différence. Vérifions la normalisation.")

    # 5. NOUVEAU TEST : Vérification de la structure tridiagonale (Hors-bande)
    masque = np.ones_like(dJ_matriciel, dtype=bool)
    for k in range(N):
        for j in range(N):
            if abs(k-j) <= 1:
                masque[k, j] = False

    erreur_hors_bande = np.max(np.abs(dJ_matriciel[masque]))
    print(f"\nErreur hors bande (doit être ~0.0) : {erreur_hors_bande:.2e}")
    
    if erreur_hors_bande < 1e-14:
         print("[SUCCÈS] La structure tridiagonale est parfaitement préservée.")
    else:
         print("[ÉCHEC] Le flot crée des termes hors-bande.")

if __name__ == "__main__":
    verifier_equations_toda(N=6)
