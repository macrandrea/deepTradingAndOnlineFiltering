import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core import KalmanFilter as KF
from core import ambiente as amb
from core import agente as agt
from core import utils as utils
import matplotlib.pyplot as plt

# Create T true price models, one for each day
def initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price, test='Almgren_and_Chriss'):
    true_price_models = []
    for _ in range(T):
        X_theta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)
        X_eta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)

        if test == 'Almgren_and_Chriss':
        
            A_theta_true_torched = torch.eye(N_assets**2, dtype=torch.float)
            A_eta_true_torched = torch.eye(N_assets**2, dtype=torch.float)

            B_theta_torched = torch.zeros(N_assets**2, N_assets) 
            B_eta_torched = torch.zeros(N_assets**2, N_assets) 

        else: 
            A_theta_true = utils.construct_decay_matrix(X_0=X_theta_0, decay_type='heterogeneous')  * 1.e-2
            A_theta_true_torched = torch.tensor(A_theta_true, dtype=torch.float)
            A_eta_true = utils.construct_decay_matrix(X_0=X_eta_0, decay_type='heterogeneous') * 1.e-2
            A_eta_true_torched = torch.tensor(A_eta_true, dtype=torch.float)

            B_theta_torched = torch.abs(torch.randn(N_assets**2, N_assets))
            B_eta_torched = torch.abs(torch.randn(N_assets**2, N_assets))  # Generate new impact matrix
        
        # Create Price model
        true_price_model = amb.PriceModel(M=N_assets, A1_init=A_theta_true_torched, B1_init=B_theta_torched,
                                    A2_init=A_eta_true_torched, B2_init=B_eta_torched,
                                    sigma_x_1=sigma_noise_impact, sigma_x_2=sigma_noise_impact, 
                                    sigma_y_1=sigma_noise_price, sigma_y_2=sigma_noise_price)
        
        true_price_models.append(true_price_model)
    
    return true_price_models

def initialize_kalman_filters(N_assets, test='Almgren_and_Chriss'):
    X_theta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)
    X_eta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)

    if test == 'Almgren_and_Chriss':
        
        A_theta_init_torched = torch.eye(N_assets**2, dtype=torch.float)
        A_eta_init_torched = torch.eye(N_assets**2, dtype=torch.float)

        B_theta_init_torched = torch.zeros(N_assets**2, N_assets) 
        B_eta_init_torched = torch.zeros(N_assets**2, N_assets) 
        
    else:
        A_theta_init = utils.construct_decay_matrix(X_0=X_theta_0, decay_type='heterogeneous')
        A_theta_init_torched = torch.tensor(A_theta_init, dtype=torch.float)
        A_eta_init = utils.construct_decay_matrix(X_0=X_eta_0, decay_type='heterogeneous')
        A_eta_init_torched = torch.tensor(A_eta_init, dtype=torch.float)

        B_theta_init_torched = torch.abs(torch.randn(N_assets**2, N_assets))
        B_eta_init_torched = torch.abs(torch.randn(N_assets**2, N_assets))  # Generate new impact matrix
        
    KF_permanent = KF.KalmanFilter(M=N_assets, A_init=A_theta_init_torched, B_init=B_theta_init_torched, Q=  torch.eye(N_assets**2), R=0.01 *torch.eye(N_assets))
    KF_temporary = KF.KalmanFilter(M=N_assets, A_init=A_eta_init_torched, B_init=B_eta_init_torched, Q=0.01 *torch.eye(N_assets**2), R=0.01 *torch.eye(N_assets))

    return KF_permanent, KF_temporary


# Other helper functions remain the same
def generate_control(agent, res_fundamental_price, res_efficient_price, remaining_inventory, time_step):
    u_t = agent.generate_control(res_fundamental_price.detach(), res_efficient_price.detach(), remaining_inventory.detach(), time_step.detach())
    return u_t

def kalman_filter_step(KF, S_tm1, u_t, N_assets):
    """
    Performs a single Kalman filter prediction.
    
    Parameters:
    - KF: The Kalman filter instance (either KF_permanent or KF_temporary).
    - S_tm1: Previous price state.
    - u_t: Current control input.
    - N_assets: Number of assets.

    Returns:
    - S_hat_t_tm1: Predicted price before the update.
    """
    # Predict the next state and covariance with the Kalman filter
    x_pred, P_pred = KF.predict(u_t)
    
    # Obtain the predicted price (before update)
    S_hat_t_tm1 = S_tm1 + torch.reshape(x_pred, (N_assets, N_assets)) @ u_t

    return S_hat_t_tm1, x_pred, P_pred

def run_daily_simulation(
    K, agent, true_price_model, S_0, KF_permanent, KF_temporary,
    residuals_fundamental_price, residuals_efficient_price, 
    inventory, N_assets, lamda=1000.
):
    """
    Runs a daily simulation for a trading agent interacting with a true price model,
    updating its state with a Kalman filter for efficient and fundamental prices.

    Parameters:
    - K: Number of time steps in the simulation.
    - agent: The trading agent generating control signals.
    - true_price_model: Model providing updates on true price returns.
    - S_0: Initial price for both efficient and fundamental prices.
    - KF_permanent: Kalman filter for fundamental price tracking.
    - KF_temporary: Kalman filter for efficient price tracking.
    - residuals_fundamental_price: Residuals for fundamental price.
    - residuals_efficient_price: Residuals for efficient price.
    - inventory: Inventory of the agent to be adjusted over time.
    - N_assets: Number of assets in the portfolio.

    Returns:
    - loss: Total loss over the simulation period.
    - daily_controls: List of control actions taken at each step.
    - S_efficient_list: Efficient price values over time.
    - S_fundamental_list: Fundamental price values over time.
    - x_predicted_fundamental_list: Predicted states for the fundamental price.
    - x_predicted_efficient_list: Predicted states for the efficient price.
    - x_filtered_fundamental_list: Filtered states for the fundamental price.
    - x_filtered_efficient_list: Filtered states for the efficient price.
    """
    # Initialize variables to track metrics and state over time
    loss = 0
    daily_controls = []
    S_efficient_list = []
    S_fundamental_list = []
    r_efficient_list = []
    r_fundamental_list = []
    S_hat_efficient_list = []
    S_hat_fundamental_list = []
    S_t_efficient = S_0
    S_t_fundamental = S_0
    residuals_S_fundamental_list = []
    residuals_S_efficient_list = []
    x_predicted_fundamental_list = []
    x_predicted_efficient_list = []
    x_filtered_fundamental_list = []
    x_filtered_efficient_list = []

    # Iterate over each time step in the daily simulation
    for k in range(K):
        # Generate control input based on residuals and the agent's strategy
        u_t = generate_control(agent, residuals_fundamental_price, residuals_efficient_price, torch.tensor(inventory), torch.tensor([k/K]))
        daily_controls.append(u_t.detach().tolist())
        true_price_model.update_control(u_t)
        #print(u_t)
        #if k >= 10:
        #    raise KeyError

        # Perform Kalman filter prediction for both efficient and fundamental prices
        S_hat_t_tm1_efficient, x_pred_efficient, P_pred_efficient = kalman_filter_step(
            KF_temporary, S_t_fundamental, u_t, N_assets
        )
        S_hat_t_tm1_fundamental, x_pred_fundamental, P_pred_fundamental = kalman_filter_step(
            KF_permanent, S_t_fundamental, u_t, N_assets
        )
        
        # Record predicted prices and states
        S_hat_efficient_list.append(S_hat_t_tm1_efficient.detach().tolist())
        S_hat_fundamental_list.append(S_hat_t_tm1_fundamental.detach().tolist())
        x_predicted_efficient_list.append(x_pred_efficient.detach().tolist())
        x_predicted_fundamental_list.append(x_pred_fundamental.detach().tolist())

        # Compute real price change using the true price model
        r_tilde_t, r_t = true_price_model.update_returns()
        S_t_efficient = S_t_efficient + r_t
        S_t_fundamental = S_t_fundamental + r_tilde_t
        S_efficient_list.append(S_t_efficient.detach().tolist())
        S_fundamental_list.append(S_t_fundamental.detach().tolist())
        r_efficient_list.append(r_tilde_t.detach().tolist())
        r_fundamental_list.append(r_t.detach().tolist())

        # Update true price model states
        true_price_model.update_states()

        # Calculate residuals between observed and predicted prices
        residuals_fundamental_price = S_t_fundamental - S_hat_t_tm1_fundamental
        residuals_efficient_price = S_t_efficient - S_hat_t_tm1_efficient
        residuals_S_efficient_list.append(residuals_efficient_price.detach().tolist())
        residuals_S_fundamental_list.append(residuals_fundamental_price.detach().tolist())

        # Update Kalman filters with real observations for both prices
        KF_permanent.update(r_t.detach(), x_pred_fundamental.detach(), P_pred_fundamental.detach(), u_t.detach())
        KF_temporary.update(r_tilde_t.detach(), x_pred_efficient.detach(), P_pred_efficient.detach(), u_t.detach())
        x_filtered_efficient = KF_temporary.x
        x_filtered_fundamental = KF_permanent.x
        x_filtered_efficient_list.append(x_filtered_efficient.detach().tolist())
        x_filtered_fundamental_list.append(x_filtered_fundamental.detach().tolist())
        
        # Compute and accumulate the loss based on the efficient price and control action
        loss += r_tilde_t @ u_t

        # Update inventory based on control action
        inventory -= u_t

    # Stack daily_controls to form a tensor of shape (K, num_assets)
    daily_controls_tensor = torch.tensor(daily_controls, dtype=torch.float)  # Shape: (K, num_assets)
    #print(daily_controls_tensor)
    # Sum across assets to get a (K,) tensor
    assets_sum = daily_controls_tensor.sum(dim=1)  # Shape: (K,)
    # Expand assets_sum to match the shape of daily_controls_tensor
    assets_sum_expanded = assets_sum.unsqueeze(1).expand_as(daily_controls_tensor)  # Shape: (K, num_assets)
    # Compute the L2 norm of the difference
    l2_norm = torch.norm(assets_sum_expanded - daily_controls_tensor, p=2)
    # Update the loss
    loss += lamda * l2_norm

    return (
        loss, daily_controls, S_efficient_list, S_fundamental_list,
        S_hat_efficient_list, S_hat_fundamental_list, x_predicted_fundamental_list, 
        x_predicted_efficient_list, x_filtered_fundamental_list, x_filtered_efficient_list, 
        r_fundamental_list, r_efficient_list
    )

def plot_fundamental_prices(list_real_price, list_predicted_price):
    """
    Plots the first and second elements of two lists in a two-row, one-column layout.

    Parameters:
    - list1: List of arrays or values for the first plot.
    - list2: List of arrays or values for the second plot.
    - title1: Title for the first plot (default: "First List").
    - title2: Title for the second plot (default: "Second List").
    """
    if len(list_real_price) != len(list_predicted_price):
        raise ValueError("Both lists must have the same length.")
    
    fundamental_prices_np = np.array(list_real_price)
    fundamental_hat_prices_np = np.array(list_predicted_price)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot the first elements of the lists
    axs[0].plot(fundamental_prices_np[:,0], label=f'real')
    axs[0].plot(fundamental_hat_prices_np[:,0], label=f'model', linestyle='--')
    axs[0].legend()
    axs[0].grid(True)

    # Plot the second elements of the lists
    axs[1].plot(fundamental_prices_np[:,1], label=f'real')
    axs[1].plot(fundamental_hat_prices_np[:,1], label=f'model', linestyle='--')
    axs[1].legend()
    axs[1].grid(True)

    # Final adjustments
    axs[1].set_xlabel("Time step")
    fig.tight_layout()
    plt.show()


def main_simulation(T=10, K=100, N_assets=2, q0=1, sigma_noise_impact=1.e-5, sigma_noise_price = 1.e-5):
    
    # Initialize inventory and previous control action
    inventory = q0 * torch.ones(N_assets)
    
    # Generate T true price models, one for each day
    true_price_models = initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price)

    # Initialize agent 
    agent = agt.Agent(q0=q0,
                      K=K, 
                      in_size=3*N_assets+1, 
                      out_size=N_assets,
                      learning_rate=1.e-5,
                      l2_radius=0.1,
                      lipschitz_constant=0.9)
    
    # Initial state variables
    S_0 = torch.ones(N_assets) * 100.
    residuals_fundamental, residuals_efficient = torch.ones(N_assets) * .1, torch.ones(N_assets) * .1

    # Storage for simulation results
    total_controls, residuals_fundamental_price_list, residuals_efficient_price_list, agents = [], [], [], []

    # Initialize Kalman Filters with random impact matrices
    KF_permanent, KF_temporary = initialize_kalman_filters(N_assets)

    # Simulation loop over time steps
    for t in range(T):
        print('new trading day')
        true_price_model = true_price_models[t]  # Select the model for the current day
        loss, daily_controls, S_efficient_list, S_fundamental_list, S_hat_efficient_list, S_hat_fundamental_list, x_predicted_fundamental_list, x_predicted_efficient_list, x_filtered_fundamental_list, x_filtered_efficient_list, r_fundamental_list, r_efficient_list = run_daily_simulation(
            K=K, 
            agent=agent, 
            true_price_model=true_price_model, 
            S_0=S_0, 
            KF_permanent=KF_permanent, 
            KF_temporary=KF_temporary, 
            residuals_fundamental_price=residuals_fundamental, 
            residuals_efficient_price=residuals_efficient, 
            inventory=inventory, 
            N_assets=N_assets
        )
        # Train agent with accumulated loss and save state
        agent.train(loss)
        agents.append(agent)

        # Create a new agent with the same parameters
        new_agent = agent = agt.Agent(q0=q0,
                      K=K, 
                      in_size=3*N_assets+1, 
                      out_size=N_assets,
                      learning_rate=agent.optimizer.param_groups[0]['lr'],
                      l2_radius=0.1,
                      lipschitz_constant=0.9)

        # Copy the neural network weights from the current agent to the new agent
        new_agent.nn.load_state_dict(agent.nn.state_dict())

        # Update agent reference to the new agent
        agent = new_agent

        S_fundamental_torch = torch.tensor(S_fundamental_list)
        S_hat_fundamental_torch = torch.tensor(S_hat_fundamental_list)

        S_efficient_torch = torch.tensor(S_efficient_list)
        S_hat_efficient_torch = torch.tensor(S_hat_efficient_list)

        residuals_fundamental = torch.mean(S_fundamental_torch - S_hat_fundamental_torch, axis=0)
        residuals_efficient = torch.mean(S_efficient_torch - S_hat_efficient_torch, axis=0)
        
        plot_fundamental_prices(S_fundamental_list, S_hat_fundamental_list)
        # Fit Kalman Filters for the next day's parameters
        KF_permanent.fit(Y=r_fundamental_list, U=daily_controls)
        KF_temporary.fit(Y=r_efficient_list, U=daily_controls)
        
        # Record daily controls
        total_controls.append(daily_controls)
        #print(f'{t + 1} day of trading has been done.')
        #print('New fitter params for KF_permanent are:')
        #print('A matrix:', KF_permanent.A)
        #print('B matrix:', KF_permanent.B)

        #print('New fitter params for KF_temporary are:')
        #print('A matrix:', KF_temporary.A)
        #print('B matrix:', KF_temporary.B)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    # Assuming total_controls is a torch tensor of shape (T, K, N_assets)
    # Convert it to a NumPy array for easier plotting with matplotlib if necessary
    total_controls_np = np.array(total_controls)


    T, K, N_assets = total_controls_np.shape

    # Create a figure with 3 rows for each asset
    fig, axes = plt.subplots(N_assets, 1, figsize=(10, 6), sharex=True)

    # Set up the x-axis
    x = np.arange(K)

    # Initialize empty lists to hold lines for each trajectory
    lines = [[axes[i].plot([], [])[0] for _ in range(T)] for i in range(N_assets)]

    # Update function for animation
    def update(day):
        for asset in range(N_assets):
            for t in range(day + 1):  # Plot up to the current day incrementally
                lines[asset][t].set_data(x, total_controls_np[t, :, asset])
                lines[asset][t].set_alpha(0.5)  # Set a semi-transparent line style for each trajectory
            axes[asset].set_ylim(
                np.min(total_controls_np[:day + 1, :, asset]) - 1,
                np.max(total_controls_np[:day + 1, :, asset]) + 1
            )  # Adjust y-axis to fit the data

        fig.suptitle(f'Day {day + 1} of Trading')
        return [line for sublist in lines for line in sublist]

    # Set up the plot limits
    for asset in range(N_assets):
        axes[asset].set_xlim(0, K)
        axes[asset].set_ylabel(f'Asset {asset + 1} Control')
    axes[-1].set_xlabel('Time Step (K)')

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(T), interval=5000, blit=True, repeat=False
    )

    plt.show()

if __name__ == "__main__":
    main_simulation()

