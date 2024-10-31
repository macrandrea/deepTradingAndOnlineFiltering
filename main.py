#%%
import ambiente as amb
import agente as agt
import numpy as np

price_0 = 0.01
T = 10 # outer time horizon
K = 10 # inner time horizon
n = 2
imp_0 = np.ones((n)) * 0.001
A_0 = np.ones((n,n)) * np.random.rand(n, n) * 0.001
B_0 = np.ones((n,n)) * np.random.rand(n, n) * 0.002
C_0 = np.ones((n,n)) * np.random.rand(n, n) * 0.003
D_0 = np.ones((n,n)) * np.random.rand(n, n) * 0.004
y_0 = np.ones((n,n)) * 0.01
u_t = np.zeros(n)

ritorni = np.zeros((n, T))
imp = []

pri = amb.Price(price_0, imp_0, A_0, B_0, C_0, D_0, y_0, u_t, N_assets = n)
age = agt.Agente(pri)

age.train()

#for t in range(T):
#    for k in range(K):
#        u_t = np.random.uniform(low=0.0, high=10.0, size=n) # output of a NN (agent)
#        ritorni[..., t], impact = pri.get_price_ret(u_t)
