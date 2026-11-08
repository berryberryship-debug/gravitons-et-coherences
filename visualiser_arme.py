import numpy as np
import matplotlib.pyplot as plt

# On charge la perturbation trouvée par l'algorithme
h_opt = np.load("h_optimal.npy")
positions = np.arange(len(h_opt))

plt.figure(figsize=(10, 4))

# Graphique 1 : La forme de l'onde
plt.subplot(1, 2, 1)
plt.stem(positions, h_opt, basefmt="k-")
plt.title("Profil de la perturbation optimale \(h_i\)")
plt.xlabel("Site \(i\)")
plt.ylabel("Amplitude")
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Graphique 2 : Le spectre de Fourier (pour voir les fréquences spatiales)
plt.subplot(1, 2, 2)
spectre = np.abs(np.fft.rfft(h_opt))
freqs = np.fft.rfftfreq(len(h_opt))
plt.plot(freqs[1:], spectre[1:], 'o-') # On ignore la fréquence 0 (moyenne)
plt.title("Spectre de Fourier (Modes spatiaux)")
plt.xlabel("Fréquence spatiale")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("arme_absolue.png")
print("Graphique sauvegardé : arme_absolue.png")
print("Max amplitude :", np.max(np.abs(h_opt)))
plt.show()
