import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core import KalmanFilter as KF
from core import ambiente as amb
from core import agente as agt
from core import utils as utils

def main_simulation(T=100, K=10, N_assets=3, q0=100, sigma_noise_impact=0.01, sigma_noise_price = 0.01):
    
    # Initialize inventory and previous control action
    inventory = q0 * torch.abs(torch.randn(N_assets))
    u_tm1 = torch.zeros(N_assets)
    

    # Generate T true price models, one for each day
    true_price_models, B_list = initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price)

    # Initialize agent
    agent = agt.Agente(K=K)
    
    # Initial state variables
    S_tm1 = np.ones((N_assets, 1)) * 100
    residuals_fundamental, residuals_efficient = 10.0, 10.0

    # Storage for simulation results
    total_controls, residuals_fundamental_price_list, residuals_efficient_price_list, agents = [], [], [], []

    # Initialize Kalman Filters with random impact matrices
    KF_permanent, KF_temporary = initialize_kalman_filters(N_assets)

    # Simulation loop over time steps
    for t in range(T):
        true_price_model = true_price_models[t]  # Select the model for the current day
        impact_tm1 = B_list[t]
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
    B_list = []
    for _ in range(T):
        X_theta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets)
        X_eta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets)

        A_theta_true = utils.construct_decay_matrix(X_0=X_theta_0, decay_type='heterogeneous')
        A_eta_true = utils.construct_decay_matrix(X_0=X_eta_0, decay_type='heterogeneous')
        
        B = torch.abs(torch.randn(N_assets, N_assets))  # Generate new impact matrix

        # Create Price model
        A = torch.stack([A_theta_true, A_eta_true])
        true_price_model = amb.Price(A, B, sigma_noise_impact, sigma_noise_price, N_assets)
        true_price_models.append(true_price_model)
        B_list.append(B)
    return true_price_models, B_list

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

        # Perform Kalman filter step for efficient price (KF_temporary)
        S_tilde_t, S_tilde_hat_t_tm1 = kalman_filter_step(KF_temporary, S_tm1, u_t, true_price_model, impact_t, N_assets, "efficient")
        S_tilde_list.append(S_tilde_t)
        residual_S_tilde = S_tilde_t - S_tilde_hat_t_tm1
        residuals_efficient_price_list.append(residual_S_tilde.item())
        loss += S_tilde_t @ u_t

        # Update inventory
        inventory -= u_t

        # Perform Kalman filter step for fundamental price (KF_permanent)
        S_t, S_hat_t_tm1 = kalman_filter_step(KF_permanent, S_tm1, u_t, true_price_model, impact_t, N_assets, "fundamental")
        S_list.append(S_t)
        residual_S = S_t - S_hat_t_tm1
        residuals_fundamental_price_list.append(residual_S.item())

        impact_tm1 = impact_t
        u_t = u_tm1

    return loss, daily_controls, S_list, S_tilde_list

def kalman_filter_step(KF, S_tm1, u_t, price_model, impact_t, N_assets, price_type):
    """
    Performs a single Kalman filter prediction and update step for either the efficient or fundamental price.
    
    Parameters:
    - KF: The Kalman filter instance (either KF_permanent or KF_temporary).
    - S_tm1: Previous price state.
    - u_t: Current control input.
    - price_model: The price model instance.
    - impact_t: The current impact state.
    - N_assets: Number of assets.
    - price_type: Type of price to handle ("efficient" or "fundamental").

    Returns:
    - S_t: Updated price after applying control and impact.
    - S_hat_t_tm1: Predicted price before the update.
    """
    # Predict the next state and covariance with the Kalman filter
    x_pred, P_pred = KF.predict(u_t)
    
    # Obtain the predicted price (before update)
    S_hat_t_tm1 = S_tm1 + x_pred.reshape(N_assets, N_assets)

    # Select eta or theta based on price type for generating the new price
    if price_type == "efficient":
        impact_component = impact_t[:N_assets**2]
    else:
        impact_component = impact_t[N_assets**2:]
    
    # Generate the new observed price
    S_t = price_model.generate_price(S_tm1, impact_component, u_t, noise=True)

    # Update the Kalman filter based on the observed price
    KF.update(S_t.ravel(), x_pred, P_pred)

    return S_t, S_hat_t_tm1

# Other helper functions remain the same
def generate_impact_and_control(agent, price_model, imp_tm1, u_tm1, res_fundamental_price, res_efficient_price):
    impact_t = price_model.evolve_hidden_state(imp_tm1, u_tm1)
    u_t = agent.generate_control(res_fundamental_price, res_efficient_price)
    return impact_t, u_t

