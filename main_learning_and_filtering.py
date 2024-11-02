import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core import KalmanFilter as KF
from core import ambiente as amb
from core import agente as agt


def main_simulation(T=100, K=10, N_assets=3, q0=100, sigma_noise_impact=0.01, sigma_noise_price = 0.01):
    
    # Initialize inventory and previous control action
    inventory = q0 * torch.abs(torch.randn(N_assets))
    u_tm1 = torch.zeros(N_assets)

    # Generate T true price models, one for each day
    true_price_models = initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price)

    # Initialize agent
    agent = agt.Agente(K=K)
    
    # Initial state variables
    S_tm1 = np.ones((N_assets, 1)) * 100
    impact_tm1 = np.random.randn(N_assets**2)
    residuals_fundamental, residuals_efficient = 10.0, 10.0

    # Storage for simulation results
    total_controls, residuals_fundamental_price_list, residuals_efficient_price_list, agents = [], [], [], []

    # Initialize Kalman Filters with random impact matrices
    KF_permanent, KF_temporary = initialize_kalman_filters(N_assets)

    # Simulation loop over time steps
    for t in range(T):
        true_price_model = true_price_models[t]  # Select the model for the current day
        
        loss, daily_controls, S_list, S_tilde_list = run_daily_simulation(
            K, agent, true_price_model, S_tm1, impact_tm1, u_tm1, KF_permanent, KF_temporary, 
            residuals_fundamental_price_list, residuals_efficient_price_list, 
            residuals_fundamental, residuals_efficient, inventory, N_assets
        )

        # Train agent with accumulated loss and save state
        agent.train(loss)
        agents.append(agent)
        agent = agent.copy()

        # Fit Kalman Filters for the next day's parameters
        KF_permanent.fit(Y=S_list, U=daily_controls)
        KF_temporary.fit(Y=S_tilde_list, U=daily_controls)
        
        # Record daily controls
        total_controls.append(daily_controls)

# Create T true price models, one for each day
def initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price):
    true_price_models = []
    for _ in range(T):
        A_true = np.random.randn(N_assets**2, N_assets**2)
        B_true = np.random.randn(N_assets**2, N_assets)
        true_price_model = amb.Price(A_true, B_true, sigma_noise_impact, sigma_noise_price, N_assets)
        true_price_models.append(true_price_model)
    return true_price_models

def initialize_kalman_filters(N_assets):
    A_theta_init = torch.diag(torch.abs(torch.randn(N_assets, N_assets)))
    A_eta_init = torch.diag(torch.abs(torch.randn(N_assets, N_assets)))
    B_theta_init = torch.abs(torch.randn(N_assets**2, N_assets))
    B_eta_init = torch.abs(torch.randn(N_assets**2, N_assets))
    KF_permanent = KF.KalmanFilterWithMLE(M=N_assets, A_init=A_theta_init, B_init=B_theta_init, Q=torch.eye(N_assets), R=torch.eye(N_assets))
    KF_temporary = KF.KalmanFilterWithMLE(M=N_assets, A_init=A_eta_init, B_init=B_eta_init, Q=torch.eye(N_assets), R=torch.eye(N_assets))
    return KF_permanent, KF_temporary

def run_daily_simulation(
    K, agent, true_price_model, S_tm1, impact_tm1, u_tm1, KF_permanent, KF_temporary,
    residuals_fundamental_price_list, residuals_efficient_price_list, 
    residuals_fundamental_price, residuals_efficient_price, inventory, N_assets
):
    loss = 0
    daily_controls = []
    S_list = []
    S_tilde_list = []
    
    for k in range(K):
        # Generate temporary impact and control
        impact_t, u_t = generate_impact_and_control(agent, true_price_model, impact_tm1, u_tm1, 
                                                    residuals_fundamental_price, residuals_efficient_price, N_assets)
        daily_controls.append(u_t)

        # Efficient price prediction and residuals computation
        S_tilde_t, S_tilde_hat_t_tm1 = compute_efficient_price(KF_temporary, u_t, S_tm1, true_price_model, impact_t, N_assets)
        S_tilde_list.append(S_tilde_t)
        residual_S_tilde = S_tilde_t - S_tilde_hat_t_tm1
        residuals_efficient_price_list.append(residual_S_tilde.item())
        loss += S_tilde_t @ u_t

        # Update inventory
        inventory -= u_t

        # Fundamental price prediction and residuals computation
        S_t, S_hat_t_tm1 = compute_fundamental_price(KF_permanent, u_t, S_tm1, true_price_model, impact_t, N_assets)
        S_list.append(S_t)
        residual_S = S_t - S_hat_t_tm1
        residuals_fundamental_price_list.append(residual_S.item())

    return loss, daily_controls, S_list, S_tilde_list

def generate_impact_and_control(agent, price_model, imp_tm1, u_tm1, res_fundamental_price, res_efficient_price, N_assets):
    impact_t = price_model.evolve_hidden_state(imp_tm1, u_tm1)
    u_t = agent.generate_control(res_fundamental_price, res_efficient_price)
    return impact_t, u_t

def compute_efficient_price(KF, u_t, S_tm1, price_model, impact_t, N_assets):
    x_eta_t_tm1, _ = KF.predict(u_t)
    S_tilde_hat_t_tm1 = S_tm1 + x_eta_t_tm1.reshape(N_assets, N_assets)
    eta_tm1 = impact_t[:N_assets**2]
    S_tilde_t = price_model.generate_price(S_tm1, eta_tm1, u_t, noise=True)
    return S_tilde_t, S_tilde_hat_t_tm1

def compute_fundamental_price(KF, u_t, S_tm1, price_model, impact_t, N_assets):
    x_theta_t_tm1, _ = KF.predict(u_t)
    S_hat_t_tm1 = S_tm1 + x_theta_t_tm1.reshape(N_assets, N_assets)
    theta_tm1 = impact_t[N_assets**2:]
    S_t = price_model.generate_price(S_tm1, theta_tm1, u_t, noise=True)
    return S_t, S_hat_t_tm1
