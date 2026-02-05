import numpy as np
import matplotlib.pyplot as plt

def u0(x):
    u0 = 2.0/(2.0 - np.cos(x))
    return u0

def V(x): # external potential
    V = 3.0/(5.0-4.0*np.cos(x))
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

def init(f, n1, n2 = None, ite = None):
    if n2 == None : 
        A = np.zeros(n1+2, dtype = complex)
        A[1:n1+1]=f
        A[0] = A[n1] # pour condition périodique : ajout factice de la dernière valeure avant la première
        A[n1+1] = A[1]
    else : 
        A = np.zeros((n1,n2+2), dtype = complex)
        if len(np.shape(f)) == 1 :
            for i in range(ite):
                A[i,1:n2+1]=f
                A[i,0] = A[i,n2]
                A[i,n2+1] = A[i,1]
        elif len(np.shape(f)) == 2 :
            for i in range(ite):
                A[i,1:n2+1]=f[i]
                A[i,0] = A[i,n2]
                A[i,n2+1] = A[i,1]
        else :
            print("f est de taille supérieur à 2")
    return A

alpha = 1.0
nx = 2**8
tau = 0.1
T = 1
nt = int(np.floor(T/tau)) + 1
x = np.linspace(0, 2*np.pi, nx, endpoint=False)
t = np.linspace(0, tau*(nt-1), nt)
dx = (2.0*np.pi)/nx

Vx = init(V(x), nx)
S = init(noyau_chaleur(tau,nx), nx)

nc = 500 # nombre de réalisations pour monte carlo
mass_splitting = np.zeros((nc, nt))
mass_duhamel = np.zeros((nc, nt))
mass_sEXP = np.zeros((nc, nt))

# Boucle Monte Carlo pour les calculs d'espérance
for i in range(nc): 
    # Initialisation
    u = init(u0(x), nt, nx, 1)
    u_duhamel = init(u0(x), nt, nx, 1)
    u_sEXP = init(u0(x), nt, nx, 1)

    W = init(Wq(nt, tau, nx, x), nt, nx, nt)

    for n in range(1, nt):
        # Schéma de splitting
        u_prev = u[n-1, :]
        u_bis = np.exp(-1j*tau*Vx)*u_prev
        u_bis = u_bis - 1j*alpha*W[n]
        u_lin = np.fft.ifft(S*np.fft.fft(u_bis))
        u[n, :] = u_lin
        u[n, 0] = u[n, nx]
        u[n, nx+1] = u[n, 1]

        # Formule de Duhamel
        temps = t[n]
        integrale1 = np.zeros(nx+2, dtype=complex)
        integrale2 = np.zeros(nx+2, dtype=complex)
        for l in range(1, n+1):
            s = tau*(l-1)
            Ss = init(noyau_chaleur(temps-s,nx), nx)
            integrale1 += np.fft.ifft(Ss*np.fft.fft(Vx*u_duhamel[l-1,:])) * tau
            integrale2 += np.fft.ifft(Ss*np.fft.fft(W[l])) * tau
        St = init(noyau_chaleur(temps,nx), nx)
        u_init = init(u0(x), nx)
        u_duhamel[n, :] = np.fft.ifft(St*np.fft.fft(u_init)) - 1j*integrale1 - 1j*alpha*integrale2
        u_duhamel[n, 0] = u_duhamel[n, nx]
        u_duhamel[n, nx+1] = u_duhamel[n, 1]

        # Schéma stochastique exponentiel
        u_sEXP_prev = u_sEXP[n-1, :]
        u_sEXP_bis = u_sEXP_prev - 1j*tau*Vx*u_sEXP_prev - 1j*alpha*W[n]
        u_sEXP_lin = np.fft.ifft(S*np.fft.fft(u_sEXP_bis))
        u_sEXP[n, :] = u_sEXP_lin
        u_sEXP[n, 0] = u_sEXP[n, nx]
        u_sEXP[n, nx+1] = u_sEXP[n, 1]

    mass_splitting[i] = dx*np.sum(np.abs(u[:, 1:nx+1])**2, axis=1)
    mass_duhamel[i] = dx*np.sum(np.abs(u_duhamel[:, 1:nx+1])**2, axis=1)
    mass_sEXP[i] = dx*np.sum(np.abs(u_sEXP[:, 1:nx+1])**2, axis=1)

E_mass_splitting = 1/nc * np.sum(mass_splitting, axis=0)
E_mass_duhamel = 1/nc * np.sum(mass_duhamel, axis=0)
E_mass_sEXP = 1/nc * np.sum(mass_sEXP, axis=0)

k = np.arange(nx)
TrQ = np.sum(gamma(k)**2)
E_mass_theorique = E_mass_splitting[0] + t*(alpha**2)*TrQ

# Implémentation du schéma déterministe (alpha = 0)
u_det = init(u0(x), nt, nx, 1)

for m in range(1, nt):
    # Schéma déterministe
    u_prev_det = u_det[m-1, :]
    u_bis_det = np.exp(-1j*tau*Vx)*u_prev_det
    u_lin_det = np.fft.ifft(S * np.fft.fft(u_bis_det))

    u_det[m, :] = u_lin_det
    u_det[m, 0] = u_det[m, nx]
    u_det[m, nx+1] = u_det[m, 1]

mass_det = dx * np.sum(np.abs(u_det[:, 1:nx+1])**2, axis=1)

erreur_L2_final = np.linalg.norm(u[-1, 1:nx+1] - u_duhamel[-1, 1:nx+1], ord=2)
print("Erreur schéma splitting et formule duhamel (t=1) :", erreur_L2_final)
print("Différence mass :", np.linalg.norm(E_mass_splitting - E_mass_theorique))

plt.subplot(221)
plt.plot(x, np.real(u[0,1:nx+1]), label="Condition initiale", linewidth=1.5)
plt.plot(x, np.real(u[-1, 1:nx+1]), "r", label="Solution splitting", linewidth=1.5)
plt.plot(x, np.real(u_det[-1, 1:nx+1]), "g", label="Solution déterministe", linewidth=1.5)
plt.plot(x, np.real(u_duhamel[-1, 1:nx+1]), "--r", label="Solution duhamel", linewidth=1.5)
plt.plot(x, np.real(u_sEXP[-1, 1:nx+1]), ":r", label="Solution sEXP", linewidth=1.5)
plt.legend(fontsize="small")
plt.title("Graphique de Real(u) en fonction de x")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.subplot(222)
plt.plot(x, np.abs(u[0,1:nx+1]), label="Condition initiale", linewidth=1.5)
plt.plot(x, np.abs(u[-1, 1:nx+1]), "r", label="Solution splitting", linewidth=1.5)
plt.plot(x, np.abs(u_det[-1, 1:nx+1]), "g", label="Solution déterministe", linewidth=1.5)
plt.plot(x, np.abs(u_duhamel[-1, 1:nx+1]), "--r", label="Solution duhamel", linewidth=1.5)
plt.plot(x, np.abs(u_sEXP[-1, 1:nx+1]), ":r", label="Solution sEXP", linewidth=1.5)
plt.legend(fontsize="small")
plt.title("Graphique de |u| en fonction de x")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.subplot(223)
plt.plot(x, np.abs(u[0, 1:nx+1]), label="Temps 0", linewidth=1.5)
plt.plot(x, np.abs(u[int(nt/2), 1:nx+1]), label="Temps "+str(int(nt/2)*tau), linewidth=1.5)
plt.plot(x, np.abs(u[-1, 1:nx+1]), label="Temps "+str(T), linewidth=1.5)
plt.legend(fontsize="small")
plt.title("Graphique de |u| splitting")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.subplot(224)
plt.plot(t, mass_det, "g", label="Cas déterministe (alpha=0)", linewidth=1.5)
plt.plot(t, E_mass_splitting, "red", label="Espérance stochastique splitting", linewidth=1.5)
plt.plot(t, E_mass_duhamel, "--r", label="Espérance stochastique duhamel", linewidth=1.5)
plt.plot(t, E_mass_sEXP, ":r", label="Espérance stochastique sEXP", linewidth=1.5)
plt.plot(t, E_mass_theorique, ":k", label="Espérance théorique", linewidth=1.5)
plt.legend(fontsize="small")
plt.title("Comparaison des espérances")
plt.xlabel("Temps")
plt.ylabel("Masse")
plt.grid(True, alpha=0.3)

plt.suptitle("tau = "+str(tau)+", nx = "+str(nx))
plt.tight_layout()
plt.show()