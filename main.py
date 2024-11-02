#%%
import core.ambiente as amb
import core.agente as agt
import numpy as np
from core import KalmanFilter as KF
import torch
# %%
# Market Initialization
N_assets = 3  # Number of assets in the market
S_0 = 100.0   # Initial price level for each asset
S_tm1 = S_0 * torch.ones((N_assets, 1))  # Initial prices as a column vector

# Generate permanent and temporary impact factors
theta_generator_diag = torch.abs(torch.randn(N_assets))  # Diagonal for permanent impact matrix
eta_generator_diag = torch.abs(torch.randn(N_assets))    # Diagonal for temporary impact matrix

# Diagonal impact matrices for Theta (permanent) and Eta (temporary)
A_theta = torch.diag(theta_generator_diag)
A_eta = torch.diag(eta_generator_diag)
A = torch.stack((A_theta, A_eta), dim=0)  # Combined 3D tensor (2, N_assets, N_assets)

# Random impact matrices with specified dimensions for B
B_theta = torch.abs(torch.randn((N_assets**2, N_assets)))
B_eta = torch.abs(torch.randn((N_assets**2, N_assets)))
B = torch.stack((B_theta, B_eta), dim=0)  # Shape: (2, N_assets**2, N_assets)

# Initial endogenous factors for Theta and Eta
theta_tm1 = torch.abs(torch.randn(N_assets**2))
eta_tm1 = torch.abs(torch.randn(N_assets**2))
impact_tm1 = torch.stack((theta_tm1, eta_tm1), dim=0)  # Initial hidden states (2, N_assets**2)

# Initialize price evolution with model parameters
price = amb.Price(A, B, sigma_noise_impact=0.01, sigma_noise_price=0.01, N_assets=N_assets)

# Lists to track controls and residuals over time
list_of_controls = []
list_of_residuals_fundamental_price = []
list_of_residuals_efficient_price = []

# Initial residuals for fundamental and efficient prices
residuals_fundamental_price = 1e1
residuals_efficient_price = 1e1

# Inventory initialization
Q = 100
inventory = Q * torch.abs(torch.randn(N_assets))
u_tm1 = torch.zeros(N_assets)  # Previous control action

T = 100

age = agt.Agente(T = T,
                 K = T)

KF_permanent = KF.KalmanFilterWithMLE(M=N_assets, 
                                      Q=torch.eye(N_assets), 
                                      R=torch.eye(N_assets)
                                      )
KF_temporary = KF.KalmanFilterWithMLE(M=N_assets, 
                                      Q=torch.eye(N_assets), 
                                      R=torch.eye(N_assets)
                                      )

# Simulation over time steps
for time_step in range(T):
    # Generate temporary impact based on previous impact and control
    impact_t = price.evolve_hidden(imp_tm1=impact_tm1, u_tm=u_tm1)
    
    # Generate the control action for the current time step
    u_t = age.get_control(residuals_fundamental_price=residuals_fundamental_price,
                          residuals_efficient_price=residuals_efficient_price)
    list_of_controls.append(u_t)

    # Efficient price prediction based on Kalman Filter (KF) model
    S_tilde_hat_t_tm1 = KF.predict_S_tilde(u_t)

    # Calculate observed efficient price for trader
    S_tilde_t = price.efficient_price(S_tm1, eta_tm1, u_t, N_assets)

    # Compute the residual error between observed and predicted efficient prices
    residual_S_tilde = S_tilde_t - S_tilde_hat_t_tm1
    list_of_residuals_efficient_price.append(residual_S_tilde.item())

    # Update inventory based on control action
    inventory -= u_t

    # Calculate fundamental price evolution
    S_t = price.fundamental_price(S_tm1, theta_tm1, u_t, N_assets)

    # Prepare control action for the next iteration
    u_tm1 = u_t.clone()


    


