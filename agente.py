import ambiente as amb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class KalmanFilter():
    def __init__(self, A, B, C, D, Q, R, P, x0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def update(self, u, z):
        # Prediction update
        x_hat = self.A @ self.x + self.B @ u
        P_hat = self.A @ self.P @ self.A.T + self.Q

        # Measurement update
        K = P_hat @ self.C.T @ np.linalg.inv(self.C @ P_hat @ self.C.T + self.R)
        self.x = x_hat + K @ (z - self.C @ x_hat)
        self.P = P_hat - K @ self.C @ P_hat

        return self.x
    
    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P

class DNNet(nn.Module):
    '''
    Fully connected NN with 30 nodes in each layer, for 5 layers using sequential layers
    '''
    def __init__(self, in_size, hidden_layers_size):
        super(DNNet, self).__init__()

        layers = []
        layers.append(nn.Linear(in_size, hidden_layers_size))
        for _ in range(5):  # Adding 5 hidden layers
            layers.append(nn.Linear(hidden_layers_size, hidden_layers_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_layers_size, 1))
        
        self.sequential_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential_layers(x)
    
class Agente():
    def __init__(self, ambiente, T, K):
        self.ambiente = ambiente
        self.kf = KalmanFilter(ambiente.A, ambiente.B, ambiente.C, ambiente.D, np.eye(ambiente.N_assets), np.eye(ambiente.N_assets), np.eye(ambiente.N_assets), np.zeros(ambiente.N_assets))
        self.nn = DNNet(ambiente.N_assets, 30)
        self.optimizer = optim.AdamW(self.nn.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.T = T
        self.K = K


    def get_control(self, residuals_fundamental_price, 
                    residuals_efficient_price):
        return self.nn(torch.tensor(self.kf.get_state()).float())
    
    def train(self):
        ritorni = np.zeros((self.ambiente.N_assets, self.T))
        imp = []

        for t in range(self.T):
            for k in range(self.K):
                u_t = self.get_control().detach().numpy()
                ritorni[..., t], impact = self.ambiente.get_price_ret(u_t)
                self.kf.update(u_t, impact)
                self.optimizer.zero_grad()
                loss = self.criterion()# inserire la loss per Implementation Shortfall
                loss.backward()
                self.optimizer.step()
                #mettere i plot forse?

        return ritorni

