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
def noyau_chaleur2(dt, x):
    return 1.0/((4*np.pi*1j*dt)**(1/2)) * np.exp(-np.abs(x)**2 / (4*1j*dt))


alpha = 1.0
nx = 2**8
tau = 0.1
T = 1
nt = int(np.floor(T/tau))
x = np.linspace(0, 2*np.pi, nx, endpoint=False)
t = np.linspace(0, tau*(nt-1), nt)

nc = 100 # nombre de réalisations pour monte carlo
mass = np.zeros((nc, nt))

for i in range(nc): # Monte Carlo
    u = np.zeros((nt, nx+2), dtype=complex)
    u[0,1:nx+1] = u0(x)
    u[0,0] = u[0,nx]
    u[0,nx+1] = u[0,1]

    Vx = V(x)
    W = Wq(nt, tau, nx, x)  # shape(nt, nx)
    S = noyau_chaleur(tau, nx)
    S2 = noyau_chaleur2(tau, nx)
    #print(np.shape(S), np.shape(S2))

    for n in range(1, nt):
        u_prev = u[n-1, 1:nx+1]
        ubis = np.exp(-1j*tau*Vx)*u_prev
        ubis = ubis - 1j*alpha*W[n]
        u_hat = np.fft.fft(ubis)
        u_lin = np.fft.ifft(S * u_hat)
    
        u[n, 1:nx+1] = u_lin
        u[n, 0] = u[n, nx]
        u[n, nx+1] = u[n, 1]

    #u_vrai = u[:, 1:nx+1]
    mass[i] = np.linalg.norm(u[:, 1:nx+1], ord=2, axis=1)**2

E_mass = 1/nc * np.sum(mass, axis=0)

# Calcul de la solution exacte pour comparaison
W = Wq(nt, tau, nx, x)
u_exacte = np.zeros((nt, nx), dtype=complex)
u_exacte[0] = u0(x)
for n in range(1, nt):
    integrale1 = 0
    integrale2 = 0
    ds = tau
    temp = t[n]
    for i in range(nt):
        s = tau*i
        integrale1 += noyau_chaleur(temp-s, nx)*V(x)*u_exacte[i]*ds
        integrale2 += noyau_chaleur(temp-s, nx)*u_exacte[i]*W[i] # *u_exacte[i] ????
        ds += tau
    u_exacte[n] = noyau_chaleur(temp, nx)*u0(x) - 1j*integrale1 - 1j*alpha*integrale2

plt.figure(1)
# plt.plot(x, np.real(u[0,1:nx+1]), x, np.real(u[-1, 1:nx+1]))
plt.plot(x, np.real(u[0,1:nx+1]), x, np.real(u[-1, 1:nx+1]), x, np.real(u_exacte[-1]), '--')
# plt.legend(["Temps initial", "Temps final"])
plt.legend(["Temps initial", "Temps final", "Solution exacte"])
plt.title("Graphique de u en fonction de x")
plt.xlabel("x")
plt.ylabel("u")

plt.figure(2)
plt.plot(t, E_mass)
plt.title("Evolution de la masse en fonction de t")

plt.tight_layout()
plt.show()