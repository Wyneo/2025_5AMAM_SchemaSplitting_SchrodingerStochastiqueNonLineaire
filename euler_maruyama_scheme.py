import numpy as np
import matplotlib.pyplot as plt

def u0(x):
    u0=np.zeros(np.size(x))
    for i in range(np.size(x)):
        u0[i]=2/(2-np.cos(x[i]))
    return u0

def laplacien(nx):
    l=np.diagflat([2 for i in range(nx)])+np.diagflat([-1 for i in range(nx-1)],1)+np.diagflat([-1 for i in range(nx-1)],-1)
    return l

def V(x): #external potential
    V=np.zeros(np.size(x))
    for i in range(np.size(x)):
        V[i]=3/(5-4*np.cos(x[i]))
    return V

def beta(k,t):
    np.random.seed(42)
    n=np.size(t)
    dt = t/n
    dW=np.sqrt(dt)*np.random.normal(0,1,n)
    W=np.cumsum(dW)
    return W

def gamma(k,t):
    return 1/(1+k*k)

def e(k,x):
    return (1/np.sqrt(2*np.pi))*np.exp(1j*k*x)

def Wq(t,x):
    Wq=np.zeros(np.size(t))
    for k in range(100):
        Wq = Wq + gamma(k,t)*beta(k,t)*e(k,x)
    return Wq

alpha = 1
nx = 2**8
tau = 0.1
T = 1
nt = int(1/0.1)
x = np.linspace(0,2*np.pi,nx)
t = np.linspace(0,T,nt)

u=np.zeros((nt,nx))
u[0] = u0(x)

for n in range(1,nt):
    u[n] = u[n-1] - 1j*np.dot(laplacien(nx),u[n-1]) - 1j*tau*np.dot(V(x),u[n-1]) - 1j*alpha*(Wq((n)*tau,x)-Wq((n-1)*tau,x))

mass = np.linalg.norm(u, ord=2, axis=1)

plt.plot(t, mass)
plt.show()