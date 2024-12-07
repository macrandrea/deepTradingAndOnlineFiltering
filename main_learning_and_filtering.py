#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core import KalmanFilter as KF
from core import ambiente as amb
from core import agente as agt
from core import utils as utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib.cm import get_cmap
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.style.use('default')
cmap = get_cmap('viridis')
folder = "Plots/"
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize":9,
    "font.size":9,
    "axes.titlesize" :9,
    'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}\usepackage[T1]{fontenc}\usepackage{tgbonum}',
    # Make the legend/label fonts a little smaller
    "legend.fontsize":9,
    "xtick.labelsize":9,
    "ytick.labelsize":9
}
def set_size(width='thesis', fraction=1, subplots=(1, 2)):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 483.6968
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in+.3, fig_height_in)
plt.rcParams.update(tex_fonts)
#columnwidth = 510
columnwidth = 483.6968 #thèse

#%%
# Create T true price models, one for each day
def initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price, test='Almgren_and_Chriss'):
    true_price_models = []
    #todo: initialising new day's impact with yesteday's impact
    for _ in range(T):
        X_theta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)
        X_eta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)

        if test == 'Almgren_and_Chriss':
        
            A_theta_true_torched = torch.eye(N_assets**2)
            A_eta_true_torched = torch.eye(N_assets**2)

            B_theta_torched = torch.zeros(N_assets**2, N_assets) 
            B_eta_torched = torch.zeros(N_assets**2, N_assets) 

        else: 
            A_theta_true = utils.construct_decay_matrix(X_0=X_theta_0, decay_type='heterogeneous')  * 1.e-1
            A_theta_true_torched = torch.tensor(A_theta_true, dtype=torch.float)
            A_eta_true = utils.construct_decay_matrix(X_0=X_eta_0, decay_type='heterogeneous') * 1.e-1
            A_eta_true_torched = torch.tensor(A_eta_true, dtype=torch.float)

            B_theta_torched = torch.abs(torch.randn(N_assets**2, N_assets)) * 1.e-3
            B_eta_torched = torch.abs(torch.randn(N_assets**2, N_assets)) * 1.e-3 # Generate new impact matrix
        
        # Create Price model
        true_price_model = amb.PriceModel(M=N_assets, A1_init=A_theta_true_torched, B1_init=B_theta_torched,
                                    A2_init=A_eta_true_torched, B2_init=B_eta_torched,
                                    sigma_x_1=sigma_noise_impact, sigma_x_2=sigma_noise_impact, 
                                    sigma_y_1=sigma_noise_price, sigma_y_2=sigma_noise_price)
        
        true_price_models.append(true_price_model)
    
    return true_price_models

def initialize_kalman_filters(N_assets, test='Almgren_and_Chriss'):
    X_theta_0 = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)
    X_eta_0   = utils.generate_matrix_with_positive_real_eigenvalues(size=N_assets**2)

    if test == 'Almgren_and_Chriss':
        
        A_theta_init_torched = torch.eye(N_assets**2)*1.e-3
        A_eta_init_torched = torch.eye(N_assets**2)*2.e-3

        B_theta_init_torched = torch.zeros(N_assets**2, N_assets) 
        B_eta_init_torched = torch.zeros(N_assets**2, N_assets) 
        
    else:
        A_theta_init = utils.construct_decay_matrix(X_0=X_theta_0, decay_type='heterogeneous') / 0.8
        A_theta_init_torched = torch.tensor(A_theta_init, dtype=torch.float)

        A_eta_init = utils.construct_decay_matrix(X_0=X_eta_0, decay_type='heterogeneous') / 0.8
        A_eta_init_torched = torch.tensor(A_eta_init, dtype=torch.float)


        B_theta_init_torched = torch.abs(torch.randn(N_assets**2, N_assets)) * 1.e-2
        B_eta_init_torched = torch.abs(torch.randn(N_assets**2, N_assets)) * 1.e-2 # Generate new impact matrix
        
    KF_permanent = KF.KalmanFilter(M=N_assets, A_init=A_theta_init_torched, B_init=B_theta_init_torched, Q=  torch.eye(N_assets**2), R=0.01 *torch.eye(N_assets))
    KF_temporary = KF.KalmanFilter(M=N_assets, A_init=A_eta_init_torched, B_init=B_eta_init_torched, Q=0.01 *torch.eye(N_assets**2), R=0.01 *torch.eye(N_assets))

    return KF_permanent, KF_temporary

# Other helper functions remain the same
def generate_control(agent, res_fundamental_price, res_efficient_price, remaining_inventory, time_step):
    u_t = agent.generate_control(res_fundamental_price.clone().detach(), res_efficient_price.clone().detach(), remaining_inventory.clone().detach(), time_step.clone().detach())
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
    inventory, N_assets, lamda=100.
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
    loss = 0.
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
    x_1_true = []
    x_2_true = []
    trading_cost = 0.
    trading_cost_TWAP = 0.
    u_TWAP = inventory * torch.ones(N_assets)/K
    price_model_TWAP = true_price_model.copy()
    q0 = inventory.clone().detach()
    sum_us = torch.tensor(0., requires_grad=True)
    # Iterate over each time step in the daily simulation
    for k in range(K):
        # Generate control input based on residuals and the agent's strategy
        #with torch.no_grad(): 
        u_t = generate_control(agent, residuals_fundamental_price, residuals_efficient_price, 1-torch.tensor(inventory).clone().detach()/q0, torch.tensor([k/K]).clone().detach())
        
        daily_controls.append(u_t.clone().detach().tolist())
        sum_us = sum_us + u_t
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
        S_hat_efficient_list.append(S_hat_t_tm1_efficient.clone().detach().tolist())
        S_hat_fundamental_list.append(S_hat_t_tm1_fundamental.clone().detach().tolist())
        x_predicted_efficient_list.append(x_pred_efficient.clone().detach().tolist())
        x_predicted_fundamental_list.append(x_pred_fundamental.clone().detach().tolist())

        # Compute real price change using the true price model
        r_tilde_t, r_t = true_price_model.update_returns()
        S_t_efficient = S_t_efficient + r_t
        S_t_fundamental = S_t_fundamental + r_tilde_t
        S_efficient_list.append(S_t_efficient.clone().detach().tolist())
        S_fundamental_list.append(S_t_fundamental.clone().detach().tolist())
        r_efficient_list.append(r_tilde_t.clone().detach().tolist())
        r_fundamental_list.append(r_t.clone().detach().tolist())

        # Compute the update of the TWAP price model
        price_model_TWAP.update_control(u_TWAP)
        price_model_TWAP.update_control(u_TWAP)
        r_tilde_TWAP_t, _ = price_model_TWAP.update_returns()
        
        # Update true price model states
        true_price_model.update_states()
        x_1_true.append(torch.reshape(true_price_model.x1_k, (N_assets,-1)).clone().detach().tolist())
        x_2_true.append(torch.reshape(true_price_model.x2_k, (N_assets,-1)).clone().detach().tolist())

        # Calculate residuals between observed and predicted prices
        residuals_fundamental_price = S_t_fundamental - S_hat_t_tm1_fundamental
        residuals_efficient_price = S_t_efficient - S_hat_t_tm1_efficient
        residuals_S_efficient_list.append(residuals_efficient_price.clone().detach().tolist())
        residuals_S_fundamental_list.append(residuals_fundamental_price.clone().detach().tolist())

        # Update Kalman filters with real observations for both prices
        #with torch.no_grad(): # todo: test with grads maybe 
        KF_permanent.update(r_t, x_pred_fundamental, P_pred_fundamental, u_t)
        KF_temporary.update(r_tilde_t, x_pred_efficient, P_pred_efficient, u_t)

        x_filtered_efficient = KF_temporary.x
        x_filtered_fundamental = KF_permanent.x
        x_filtered_efficient_list.append(x_filtered_efficient.clone().detach().tolist())
        x_filtered_fundamental_list.append(x_filtered_fundamental.clone().detach().tolist())
        
        # Compute and accumulate the loss based on the efficient price and control action
        loss += r_tilde_t @ u_t
        trading_cost_TWAP += (r_tilde_TWAP_t @ u_TWAP).clone().detach().item()

        # Update inventory based on control action
        inventory += u_t
    trading_cost += loss.clone().detach().item()
    # Stack daily_controls to form a tensor of shape (K, num_assets)
    daily_controls_tensor = torch.tensor(daily_controls)  # Shape: (K, num_assets)
    #print(daily_controls_tensor)
    # Sum across assets to get a (K,) tensor
    assets_sum = daily_controls_tensor.sum(dim=0)  # Shape: (K,)
    # Expand assets_sum to match the shape of daily_controls_tensor
    #assets_sum_expanded = assets_sum.unsqueeze(1).expand_as(daily_controls_tensor)  # Shape: (K, num_assets)
    # Compute the L2 norm of the difference
    q0 = torch.zeros_like(q0) + agent.q0
    q0.requires_grad = True
    l2_norm = torch.norm(sum_us - q0, p=2)
    # Update the loss
    loss += lamda * l2_norm
    #print(f"loss.requires_grad: {loss.requires_grad}")  # Should be True
    

    return (
        trading_cost, trading_cost_TWAP, loss, daily_controls, S_efficient_list, S_fundamental_list,
        S_hat_efficient_list, S_hat_fundamental_list, x_predicted_fundamental_list, 
        x_predicted_efficient_list, x_filtered_fundamental_list, x_filtered_efficient_list, 
        r_fundamental_list, r_efficient_list, x_1_true, x_2_true 
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
    fig.suptitle(f'Price per day of Trading')
    fig.tight_layout()
    plt.show()


def plot_model_and_real_prices_with_controls(real_prices, model_prices, controls, K):
    """
    Plots real and model prices along with controls iteratively over time steps.

    Parameters:
    - real_prices: np.ndarray, shape (T, K, num_assets), time series of real prices.
    - model_prices: np.ndarray, shape (T, K, num_assets), time series of model prices.
    - controls: np.ndarray, shape (T, K, num_assets), time series of controls.

    This function plots real and model prices on the left y-axis and controls on the right y-axis.
    Additionally, a horizontal line at control value 100 is added.
    """
    T, K, num_assets = controls.shape

    for t in range(T):
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(K)  # Assume assets are indexed sequentially

        # Plot real and model prices on the left y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Assets")
        ax1.set_ylabel("Prices", color='tab:blue')
        ax1.plot(x_axis, real_prices[t, :, :], label=f"Real Price,", linestyle='-', marker='o')
        ax1.plot(x_axis, model_prices[t, :, :], label=f"Model Price,", linestyle='--', marker='x')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Plot controls on the right y-axis
        ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
        ax2.set_ylabel("Controls", color='tab:red')
        
        ax2.plot(x_axis, controls[t, :, :], label=f"Control", linestyle='-.', marker='s', alpha=0.7)
        ax2.axhline(y=-1/K, color='tab:green', linestyle='--', label="Control Threshold (100)")  # Horizontal line
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(-0.2, 0.2)
        # Title and grid
        plt.title(f"Time Step {t+1}")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()

def plot_metrics(losses, trading_costs, trading_costs_TWAP):
    """
    Plots metrics in a one-row, three-column layout:
    1. Losses (log-scale y-axis)
    2. Trading costs over time
    3. Histogram comparison of trading costs
    
    Args:
        losses: List or array of loss values.
        trading_costs: List or array of trading costs.
        trading_costs_TWAP: List or array of TWAP trading costs.
    """
    fig, axes = plt.subplots(1, 3, figsize=set_size(width='thesis', subplots=(1, 3)))

    # Plot 1: Losses with log-scale on y-axis
    axes[0].plot(losses, label="Losses", color="navy")
    axes[0].set_yscale("log")
    axes[0].set_title("Losses (log-scale)")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Loss")
    #axes[0].legend()

    # Plot 2: Trading costs over time
    axes[1].plot(trading_costs, label="Trading Costs", color="navy", alpha=0.5)
    axes[1].plot(trading_costs_TWAP, label="TWAP Costs", color="black", alpha=0.5)
    axes[1].set_title("Trading Costs Over Time")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cost")
    #axes[1].legend()

    # Plot 3: Histogram of trading costs
    axes[2].hist(trading_costs, bins=50, color="navy", alpha=0.5, label="Trading Costs")
    axes[2].hist(trading_costs_TWAP, bins=50, color="black", alpha=0.5, label="TWAP Costs")
    axes[2].set_title("Histogram of Trading Costs")
    axes[2].set_xlabel("Cost")
    axes[2].set_ylabel("Frequency")
    #axes[2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("metrics_AC.pdf", transparent=True)
    plt.show()


def main_simulation(T=2, K=1000, N_assets=2, q0=1, sigma_noise_impact=1.e-4, sigma_noise_price = 1.e-4):
    
    
    # Generate T true price models, one for each day
    true_price_models = initialize_true_price_models(T, N_assets, sigma_noise_impact, sigma_noise_price, test=None)#'Almgren_and_Chriss')

    # Initialize agent 
    agent = agt.Agent(q0=q0,
                      K=K, 
                      in_size=3*N_assets+1, 
                      out_size=N_assets,
                      learning_rate=1.e-3,
                      l2_radius=0.1,
                      lipschitz_constant=0.8,
                      step_size=75, 
                      gamma=0.8)
    
    # Initial state variables
    S_0 = torch.ones(N_assets) * 1.
    residuals_fundamental, residuals_efficient = torch.ones(N_assets) * .1, torch.ones(N_assets) * .1

    # Storage for simulation results
    total_controls, total_S_fundamental, total_S_hat_fundamental, total_x1, total_x2, agents = [], [], [], [], [], []

    trading_costs, trading_costs_TWAP = [], []

    # Initialize Kalman Filters with random impact matrices
    KF_permanent, KF_temporary = initialize_kalman_filters(N_assets, test=None)#test = 'Almgren_and_Chriss')
    losses = []
    # Simulation loop over time steps
    for t in range(T):
        # Initialize inventory and previous control action
        inventory = q0 * torch.ones(N_assets)
        print(f"Episode {t + 1}/{T}: Starting new trading day")        
        true_price_model = true_price_models[t]  # Select the model for the current day
        trading_cost, trading_cost_TWAP, loss, daily_controls, S_efficient_list, S_fundamental_list, S_hat_efficient_list, S_hat_fundamental_list, x_predicted_fundamental_list, x_predicted_efficient_list, x_filtered_fundamental_list, x_filtered_efficient_list, r_fundamental_list, r_efficient_list, x_1_true, x_2_true = run_daily_simulation(
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

        # Print learning rate
        print(f"Day {t + 1}: Current LR = {agent.get_lr():.6f}")

        # Train the agent with the loss and save state
        agent.train(loss)
        agents.append(agent)
        losses.append(loss.item())
        trading_costs.append(trading_cost)
        trading_costs_TWAP.append(trading_cost_TWAP)
        
        # Print episode loss
        print(f"Episode {t + 1}/{T}: Loss = {loss:.4f}")

        # Create a new agent with the same parameters
        #new_agent = agt.Agent(q0=q0,
        #              K=K, 
        #              in_size=3*N_assets+1, 
        #              out_size=N_assets,
        #              learning_rate=agent.optimizer.param_groups[0]['lr'],
        #              l2_radius=0.1,
        #              lipschitz_constant=0.9)

        # Copy the neural network weights from the current agent to the new agent
        #new_agent.nn.load_state_dict(agent.nn.state_dict())

        # Update agent reference to the new agent
        agent = agent.clone()
        

        S_fundamental_torch = torch.tensor(S_fundamental_list)
        S_hat_fundamental_torch = torch.tensor(S_hat_fundamental_list)

        S_efficient_torch = torch.tensor(S_efficient_list)
        S_hat_efficient_torch = torch.tensor(S_hat_efficient_list)

        residuals_fundamental = (S_fundamental_torch - S_hat_fundamental_torch)[-1]
        residuals_efficient = (S_efficient_torch - S_hat_efficient_torch)[-1]
        
        #plot_fundamental_prices(S_fundamental_list, S_hat_fundamental_list)
        # Fit Kalman Filters for the next day's parameters
        KF_permanent.fit(Y=r_fundamental_list, U=daily_controls)
        KF_temporary.fit(Y=r_efficient_list, U=daily_controls)
        
        # Record daily controls
        total_controls.append(daily_controls)
        #print(daily_controls)
        total_S_fundamental.append(S_fundamental_list)
        total_S_hat_fundamental.append(S_hat_fundamental_list)

        total_x1.append(x_1_true)
        total_x2.append(x_2_true)

        #print(daily_controls)
        #print(f'{t + 1} day of trading has been done.')
        #print('New fitter params for KF_permanent are:')
        #print('A matrix:', KF_permanent.A)
        #print('B matrix:', KF_permanent.B)

        #print('New fitter params for KF_temporary are:')
        #print('A matrix:', KF_temporary.A)
        #print('B matrix:', KF_temporary.B)

    # Assuming total_controls is a torch tensor of shape (T, K, N_assets)
    # Convert it to a NumPy array for easier plotting with matplotlib if necessary
    total_controls_np = np.array(total_controls)
    total_S_fundamental_np = np.array(total_S_fundamental)
    total_S_hat_fundamental_np = np.array(total_S_hat_fundamental)

    #total_x1_np = np.array(total_x1)
    #total_x2_np = np.array(total_x2)

    #print(total_x1_np)
    #print(total_x2_np)

    #T, K, N_assets = total_controls_np.shape
    
    # Assuming total_controls_np is a 3D array with shape (T, K, N_assets)
    # Example:
    # total_controls_np = np.random.rand(T, K, N_assets)  # Replace this with your data

    plot_model_and_real_prices_with_controls(total_S_fundamental_np[-5:,:,:], total_S_hat_fundamental_np[-5:,:,:], total_controls_np[-5:,:,:], K=K)

    plot_metrics(losses, trading_costs, trading_costs_TWAP)

#if __name__ == "__main__":
#%%
# Run the main simulation
main_simulation(T=500, K=10, N_assets=2, q0=-1)
# %%
