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

def e(k,x):
    # k shape(K,), x shape(nx,) -> returns(K, nx)
    return (1.0/np.sqrt(2.0*np.pi)) * np.exp(1j * np.outer(k,x))

def Wq(nt, tau, nx, x):
    k = np.arange(nx)
    beta_k = beta(nt, nx, tau)  # shape(nt, nx)
    g = gamma(k)  # shape(nx,)
    E = e(k, x)  # shape(nx, nx)
    W = np.sum(beta_k[:, :, None] * g[None, :, None] * E[None, :, :], axis=1)
    return W

alpha = 1
nx = 2**8
tau = 0.1
T = 1
nt = int(np.floor(T/tau)) + 1
x = np.linspace(0, 2*np.pi, nx, endpoint=False)
t = np.linspace(0, tau*(nt-1), nt)
dx = (2.0*np.pi)/nx

Vx = V(x)

nc = 500 # nombre de réalisations pour monte carlo
mass = np.zeros((nc,nt))

# Boucle Monte Carlo pour les calculs d'espérance
for i in range(nc): 
    # Initialisation
    u = np.zeros((nt,nx+2), dtype=complex)
    u[0,1:nx+1] = u0(x)
    u[0,0] = u[0,nx] # pour condition périodique : ajout factice de la dernière valeure avant la première
    u[0,nx+1] = u[0,1]

    W = Wq(nt, tau, nx, x) # shape(nt, nx)

    for n in range(1,nt):
        # Schéma de Euler-Maruyama
        u_prev = u[n-1, 1:nx+1]

        u[n, 1:nx+1] = u_prev - 1j*np.dot(laplacien(nx),u_prev) - 1j*tau*np.dot(Vx,u_prev) - 1j*alpha*W[n]

        u[n, 0] = u[n, nx] # pour périodicité
        u[n, nx+1] = u[n, 1]
    
    mass[i] = dx*np.sum(np.abs(u[:, 1:nx+1])**2, axis=1)

E_mass = 1/nc * np.sum(mass, axis=0)

#E(M) théorique 
k = np.arange(nx)
TrQ = np.sum(gamma(k)**2)
E_mass_theorique = E_mass[0] + t*(alpha**2)*TrQ

plt.subplot(221)
plt.plot(x, np.real(u[0, 1:nx+1]), label="Temps initial", linewidth=1.5)
plt.plot(x, np.real(u[-1, 1:nx+1]), label="Temps final", linewidth=1.5)
plt.legend(fontsize="small")
plt.title("Graphique de Real(u) en fonction de x")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.subplot(222)
plt.plot(x, np.abs(u[0, 1:nx+1]), label="Temps initial", linewidth=1.5)
plt.plot(x, np.abs(u[-1, 1:nx+1]), label="Temps final", linewidth=1.5)
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
plt.title("Graphique de |u| au fil de t")
plt.xlabel("x")
plt.ylabel("u")
plt.grid(True, alpha=0.3)

plt.subplot(224)
plt.plot(t, E_mass, "+r", label="Espérance Euler-Maruyama", linewidth=1.5)
plt.plot(t, E_mass_theorique, ":k", label="Espérance théorique", linewidth=1.5)
plt.legend(fontsize="small")
plt.title("Comparaison des espérances")
plt.xlabel("Temps")
plt.ylabel("Masse")
plt.grid(True, alpha=0.3)

plt.suptitle("Schéma d'Euler Maruyama, tau = "+str(tau)+", nx = "+str(nx))
plt.tight_layout()
plt.show()