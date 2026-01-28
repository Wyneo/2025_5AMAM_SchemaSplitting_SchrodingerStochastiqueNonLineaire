import numpy as np
import matplotlib.pyplot as plt

def u0(x):
    u0 = 2.0/(2.0 - np.cos(x))
    return u0

def laplacien(nx):
    l=np.diagflat([2 for i in range(nx)])+np.diagflat([-1 for i in range(nx-1)],1)+np.diagflat([-1 for i in range(nx-1)],-1)
    l=l/(2*np.pi/nx)**2
    return l

def V(x): # external potential
    V=np.zeros(np.shape(x)[0])
    for i in range(np.shape(x)[0]):
        V[i]=3.0/(5.0-4.0*np.cos(x[i]))
    return V

def beta(nt, nx, tau):
    dB = np.random.normal(0, np.sqrt(tau), size=(nt, nx))
    dB[0, :] = 0.0
    return dB

def gamma(k):
    return 1.0/(1.0 + (k**2))

def e(k, x):
    # k shape(K,), x shape(nx,) -> returns(K, nx)
    return (1.0/np.sqrt(2.0*np.pi)) * np.exp(1j * np.outer(k, x))

def Wq(nt, tau, nx, x):
    k = np.arange(nx)
    beta_k = beta(nt, nx, tau)  # shape(nt, nx)
    g = gamma(k)  # shape(nx,)
    E = e(k, x)  # shape(nx, nx)
    W = np.sum(beta_k[:, :, None] * g[None, :, None] * E[None, :, :], axis=1)
    return W

def noyau_chaleur(dt, nx):
    k = np.fft.fftfreq(nx, d=(2.0*np.pi)/nx) * 2.0*np.pi
    return np.exp(-1j * (k**2) * dt)

alpha = 1.0
nx = 2**8
tau = 0.1
T = 1 #2
nt = int(np.floor(T/tau)) + 1
x = np.linspace(0, 2*np.pi, nx, endpoint=False)
t = np.linspace(0, tau*(nt-1), nt)

Vx = V(x)
S = noyau_chaleur(tau, nx)

nc = 500 # nombre de réalisations pour monte carlo
mass_splitting = np.zeros((nc, nt))
mass_duhamel = np.zeros((nc, nt))

# Boucle Monte Carlo pour les calculs d'espérance
for i in range(nc): 
    # Initialisation
    u = np.zeros((nt, nx+2), dtype=complex)
    u[0,1:nx+1] = u0(x)
    u[0,0] = u[0,nx]
    u[0,nx+1] = u[0,1]

    u_duhamel = np.zeros((nt, nx+2), dtype=complex) # Calcul de la solution exacte (formule de Duhamel)
    u_duhamel[0,1:nx+1] = u0(x)
    u_duhamel[0,0] = u_duhamel[0,nx]
    u_duhamel[0,nx+1] = u_duhamel[0,1]

    W = Wq(nt, tau, nx, x) # shape(nt, nx)

    for n in range(1, nt):
        # Schéma de splitting
        u_prev = u[n-1, 1:nx+1]
        u_bis = np.exp(-1j*tau*Vx)*u_prev - 1j*alpha*W[n]
        u_lin = np.fft.ifft(S * np.fft.fft(u_bis))
        u[n, 1:nx+1] = u_lin
        u[n, 0] = u[n, nx]
        u[n, nx+1] = u[n, 1]

        # Formule de Duhamel
        temps = t[n]
        integrale1 = np.zeros(nx, dtype=complex)
        integrale2 = np.zeros(nx, dtype=complex)
        for j in range(1, n+1):
            s = tau*j
            integrale1 += np.fft.ifft(noyau_chaleur(temps-s, nx)*np.fft.fft(Vx*u_duhamel[j,1:nx+1])) * tau
            integrale2 += np.fft.ifft(noyau_chaleur(temps-s, nx)*np.fft.fft(W[j])) * tau
        u_duhamel[n, 1:nx+1] = np.fft.ifft(noyau_chaleur(temps, nx)*np.fft.fft(u0(x))) - 1j*integrale1 - 1j*alpha*integrale2
        u_duhamel[n, 0] = u_duhamel[n, nx]
        u_duhamel[n, nx+1] = u_duhamel[n, 1]

    mass_splitting[i] = np.linalg.norm(u[:, 1:nx+1], ord=2, axis=1)**2
    mass_duhamel[i] = np.linalg.norm(u_duhamel[:, 1:nx+1], ord=2, axis=1)**2

E_mass_splitting = 1/nc * np.sum(mass_splitting, axis=0)
E_mass_duhamel = 1/nc * np.sum(mass_duhamel, axis=0)

# Implémentation du schéma déterministe (alpha = 0)
u_det = np.zeros((nt, nx+2), dtype=complex)
u_det[0,1:nx+1] = u0(x)
u_det[0,0] = u_det[0,nx]
u_det[0,nx+1] = u_det[0,1]

for n in range(1, nt):
    # Schéma déterministe
    u_prev_det = u_det[n-1, 1:nx+1]
    u_bis_det = np.exp(-1j*tau*Vx)*u_prev_det
    u_lin_det = np.fft.ifft(S * np.fft.fft(u_bis_det))

    u_det[n, 1:nx+1] = u_lin_det
    u_det[n, 0] = u_det[n, nx]
    u_det[n, nx+1] = u_det[n, 1]

mass_det = np.linalg.norm(u_det[:, 1:nx+1], ord=2, axis=1)**2

# # Test 1 : Vérification de la conservation de la masse dans le cas déterministe
# print("\nTest 1 : Conservation de la masse dans le cas déterministe")
# print("-" * 40)
# print(f"Masse initiale (déterministe) : {mass_det[0]:.6e}")
# print(f"Masse finale (déterministe)   : {mass_det[-1]:.6e}")
# print(f"Variation relative            : {abs(mass_det[-1] - mass_det[1]) / abs(mass_det[1]) * 100:.4f}%")

# # Test 2: Comparaison stochastique vs déterministe
# print("\nTest 2 : Comparaison stochastique vs déterministe")
# print("-" * 40)
# print(f"E[Masse finale] (stochastique) : {E_mass_splitting[-1]:.6e}")
# print(f"Masse finale (déterministe)    : {mass_det[-1]:.6e}")
# print(f"Différence relative            : {abs(E_mass_splitting[-1] - mass_det[-1]) / abs(mass_det[-1]) * 100:.4f}%")

# # Test 3: Vérifier que les deux méthodes donnent le même résultat en l'absence de bruit
# print("\nTest 3 : Vérification de stabilité de la masse")
# print("-" * 40)
# print(f"E[Masse initiale] (stochastique) : {E_mass_splitting[0]:.6e}")
# print(f"E[Masse finale] (stochastique)   : {E_mass_splitting[-1]:.6e}")
# print(f"Variation E[M] relative          : {abs(E_mass_splitting[-1] - E_mass_splitting[0]) / abs(E_mass_splitting[0]) * 100:.4f}%")

# # Test 4: Visualiser la solution
# print("\nTest 4 : Comparaison des solutions")
# print("-" * 40)
# erreur_L2_final = np.linalg.norm(u[-1, 1:nx+1] - u_duhamel[-1, 1:nx+1], ord=2)
# print(f"Erreur L² entre schéma et solution exacte : {erreur_L2_final:.6e}")

plt.figure(1)
plt.plot(x, np.real(u[0,1:nx+1]), label="Condition initiale", linewidth=1.5)
plt.plot(x, np.real(u[-1, 1:nx+1]), "red", label="Solution splitting", linewidth=1.5)
plt.plot(x, np.real(u_det[-1, 1:nx+1]), "--r", label="Solution déterministe", linewidth=1.5)
plt.plot(x, np.real(u_duhamel[-1, 1:nx+1]), "green", label="Solution duhamel", linewidth=1.5)
plt.legend()
plt.title("Graphique de u en fonction de x")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.figure(2)
plt.plot(t, mass_det, "--r", label="Cas déterministe (alpha=0)", linewidth=1)
plt.plot(t, E_mass_splitting, "red", label="Espérance stochastique Splitting", linewidth=1.5)
plt.plot(t, E_mass_duhamel, "green", label="Espérance stochastique Duhamel", linewidth=1.5)
plt.xlabel("Temps")
plt.ylabel("Masse L2")
plt.title("Comparaison des espérances") #  : Cas déterministe vs stochastique
plt.legend()
plt.grid(True, alpha=0.3)

plt.figure(3)
plt.plot(x, np.real(u[0, 1:nx+1]), label="Temps 0", linewidth=1.5)
plt.plot(x, np.real(u[5, 1:nx+1]), label="Temps "+str(5*tau), linewidth=1.5)
plt.plot(x, np.real(u[10, 1:nx+1]), label="Temps "+str(10*tau), linewidth=1.5)
# plt.plot(x, np.real(u[15, 1:nx+1]), label="Temps "+str(15*tau), linewidth=1.5)
# plt.plot(x, np.real(u[20, 1:nx+1]), label="Temps "+str(20*tau), linewidth=1.5)
plt.legend()
plt.title("Graphique de u au fil de t")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()