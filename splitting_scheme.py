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
        V[i]=3/(5-4*np.cos(x[i]))
    return V

def beta(nt, nx, tau, seed=42):
    rng = np.random.default_rng(seed)
    dB = rng.normal(0, np.sqrt(tau), size=(nt, nx))
    dB[0, :] = 0.0
    return dB

def gamma(k):
    return 1.0/(1.0 + (k**2))

def e(k, x):
    # k shape (K,), x shape (nx,) -> returns (K, nx)
    return (1.0/np.sqrt(2.0*np.pi)) * np.exp(1j * np.outer(k, x))

def Wq(nt, tau, x, seed=42):
    nx = np.shape(x)[0]
    k = np.arange(nx)
    beta_k = beta(nt, nx, tau, seed=seed)  # shape (nt, nx)
    g = gamma(k)  # shape (nx,)
    E = e(k, x)  # shape (nx, nx)
    W = np.sum(beta_k[:, :, None] * g[None, :, None] * E[None, :, :], axis=1)
    return W

def noyau_chaleur(dt, x):
    nx = np.shape(x)[0]
    k = np.fft.fftfreq(nx, d=(2.0*np.pi)/nx) * 2.0*np.pi
    return np.exp(-1j * (k**2) * dt)

alpha = 1.0
nx = 2**8
tau = 0.1
T = 1.0
nt = int(np.floor(T/tau)) + 1
x = np.linspace(0, 2*np.pi, nx, endpoint=False)
t = np.linspace(0, tau*(nt-1), nt)

seed = 42

W = Wq(nt, tau, x, seed=seed)  # shape (nt, nx)

u = np.zeros((nt, nx+2), dtype=complex)
u[0,1:nx+1] = u0(x)
u[0,0] = u[0,nx]
u[0,nx+1] = u[0,1]

Vx = V(x)
S = noyau_chaleur(tau, x)

for n in range(1, nt):
    u_prev = u[n-1, 1:nx+1]
    ubis = np.exp(-1j * tau * Vx * u_prev) * u_prev # Attention !!
    ubis = ubis - 1j * alpha * W[n]
    u_hat = np.fft.fft(ubis)
    u_lin = np.fft.ifft(S * u_hat)
    
    u[n, 1:nx+1] = u_lin
    u[n, 0] = u[n, nx]
    u[n, nx+1] = u[n, 1]

u_vrai = u[:, 1:nx+1]
dx = 2*np.pi/nx
mass = np.sum(np.abs(u_vrai)**2, axis=1) * dx

plt.figure(1)
plt.plot(x, np.real(u_vrai[0]), x, np.real(u_vrai[-1]))
plt.legend(["Temps initial", "Temps final"])
plt.title("Graphique de u en fonction de x")
plt.xlabel("x")
plt.ylabel("u")

plt.figure(2)
plt.plot(t, mass)
plt.title("Evolution de la masse en fonction de t")

plt.tight_layout()
plt.show()