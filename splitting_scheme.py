import numpy as np
import matplotlib.pyplot as plt

def u0(x):
    u0=np.zeros(np.size(x))
    for i in range(np.size(x)):
        u0[i]=2/(2-np.cos(x[i]))
    return u0

def laplacien(nx):
    l=np.diagflat([2 for i in range(nx)])+np.diagflat([-1 for i in range(nx-1)],1)+np.diagflat([-1 for i in range(nx-1)],-1)
    l=l/(2*np.pi/nx)**2
    return l

def V(x): #external potential
    V=np.zeros(np.size(x))
    for i in range(np.size(x)):
        V[i]=3/(5-4*np.cos(x[i]))
    return V

def beta(t):
    np.random.seed(42)
    n=np.size(t)
    dt = t/n
    dW = np.sqrt(dt)*np.random.normal(0,1,n)
    #W=np.cumsum(dW)
    return dW

def gamma(k):
    return 1/(1+k*k)

def e(k,x):
    return (1/np.sqrt(2*np.pi))*np.exp(1j*k*x)

def Wq(t,x):
    Wq=np.zeros(np.size(t))
    for k in range(100):
        Wq = Wq + gamma(k)*np.dot(beta(t),e(k,t))
    return Wq

def noyau_chaleur(t,x):
    return np.exp(-1j*t*laplacien(np.size(x)+2))
    #return np.exp(-1j*x/(2*t))/np.sqrt(2*np.pi*t)

alpha = 1
nx = 100
tau = 0.1
T = 1
nt = int(T/tau)
x = np.linspace(0,2*np.pi,nx)
t = np.linspace(0,T,nt)

u=np.zeros((nt,nx+2))
u[0,1:nx+1] = u0(x)
u[0,0] = u[0,nx] #pour condition périodique : ajout factice de la dernière valeure avant la première
u[0,nx+1] = u[0,1]
u_vrai = np.zeros((nt,nx))

incrementWiener = Wq(t,x)
incrementWiener = np.append(incrementWiener[nt-1],incrementWiener)
incrementWiener = np.append(incrementWiener,incrementWiener[1])

for n in range(1,nt):
    v = np.append(V(x)[nx-1],V(x))
    v = np.append(v,V(x)[0])

    ubis = np.exp(-1j*tau*v*u[n-1])*u[n-1]

    S = noyau_chaleur(n*tau,x)
    u[n] = np.dot(np.fft.fft(S),np.fft.fft(ubis-1j*alpha*incrementWiener[n]))

    u[n,0] = u[n,nx] #pour périodicité
    u[n,nx+1] = u[n,1]

u_vrai = np.delete(u,[0,nx+1],1) #on garde pas valeures fictives

plt.figure(1)
plt.plot(x,u_vrai[0],x,u_vrai[nt-1])
plt.legend(["Temps initial", "Temps final"])
plt.title("Graphique de u en fonction de x")
plt.xlabel("x")
plt.ylabel("u")

mass = np.linalg.norm(u_vrai, ord=2, axis=1)**2

plt.figure(2)
plt.plot(t,mass)
plt.title("Evolution de la masse en fonction de t")

plt.show()