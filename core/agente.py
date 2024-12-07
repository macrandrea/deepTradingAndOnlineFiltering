import core.ambiente as amb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm

def base_control(q0, K):
    """
    Defines the known control function u0(x).

    Parameters:
    - q0: Initial state or parameter for the control function.
    - K: Scaling factor for control.

    Returns:
    - torch.Tensor: Base control value.
    """
    return torch.tensor([-q0 / K])


class DNNet(nn.Module):
    """
    A deep neural network that produces a control signal as a perturbation of a base control function.
    The network is Lipschitz-bounded with a specified constant.
    """
    def __init__(self, q0, K, in_size, out_size, hidden_size=10, num_layers=2, lipschitz_constant=1.0, base_control_fn=None):
        super(DNNet, self).__init__()
        self.q0 = q0
        self.K = K
        self.lipschitz_constant = lipschitz_constant

        # Base control function
        self.base_control_fn = base_control_fn if base_control_fn is not None else base_control

        # Construct neural network layers with spectral normalization
        layers = [nn.Linear(in_size, hidden_size, bias=True), nn.LeakyReLU(negative_slope=0.1)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size, bias=True), nn.LeakyReLU(negative_slope=0.1)])
        
        # Add final output layer with spectral normalization
        layers.extend([nn.Linear(hidden_size, out_size, bias=False), nn.Tanh()])

        # Scale the output to enforce the Lipschitz constant
        #self.scaling_layer = nn.Linear(out_size, out_size, bias=False)
        #nn.init.constant_(self.scaling_layer.weight, self.lipschitz_constant)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Computes the control as the sum of the base control and neural network perturbation.

        Parameters:
        - x: Input tensor.

        Returns:
        - torch.Tensor: Control signal.
        """
        # Compute base control
        base_control = self.base_control_fn(self.q0, self.K)
        
        # Compute neural network perturbation and scale it
        perturbation = self.model(x)
        
        # Combine base control and perturbation
        return perturbation # + base_control  #perturbation + 

class Agent:
    def __init__(self, q0, K, in_size, out_size, learning_rate=0.001, l2_radius=0.9, lipschitz_constant=1.0, step_size=50, gamma=0.5):
        """
        Initializes the agent with a neural network, optimizer, and learning rate scheduler.

        Args:
            q0: Initial parameter for the neural network.
            K: Number of control steps.
            in_size: Input size for the neural network.
            out_size: Output size for the neural network.
            learning_rate: Initial learning rate for the optimizer.
            l2_radius: L2 regularization radius.
            lipschitz_constant: Lipschitz constant for the neural network.
            step_size: Number of steps before reducing the learning rate.
            gamma: Multiplicative factor for learning rate decay.
        """
        self.nn = DNNet(q0=q0, K=K, in_size=in_size, out_size=out_size, lipschitz_constant=lipschitz_constant)
        self.optimizer = optim.AdamW(self.nn.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.criterion = nn.MSELoss()
        self.K = K
        self.q0 = q0
        self.step_count = 0
        self.l2_radius = l2_radius
        self.in_size = in_size
        self.out_size = out_size

    def generate_control(self, residuals_fundamental, residuals_efficient, remaining_inventory, time_step):
        """
        Generates a control signal based on the input features.

        Parameters:
        - residuals_fundamental: Tensor of fundamental residuals.
        - residuals_efficient: Tensor of efficient residuals.
        - remaining_inventory: Tensor of remaining inventory.
        - time_step: Tensor representing the time step.

        Returns:
        - torch.Tensor: Control signal.
        """
        self.step_count += 1
        input_data = torch.cat([residuals_fundamental, residuals_efficient, remaining_inventory, time_step], dim=0).detach()
        return self.nn(input_data)

    def train(self, loss):
        if self.step_count >= self.K:
            self.optimizer.zero_grad()
            
            # Debug loss
            #print(f"Loss grad_fn: {loss.grad_fn}")
            
            # Backward pass
            with torch.autograd.set_detect_anomaly(True):
                try:
                    loss.backward()
                except RuntimeError as e:
                    print("Error during backward:", e)
                    raise
                
            # Optimizer step
            self.optimizer.step()  # Update parameters
            self.scheduler.step()  # Update the learning rate
            # self.project_weights_to_l2_ball()

    def get_lr(self):
        """
        Returns the current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
    
    def project_weights_to_l2_ball(self):
        """
        Projects all network parameters onto an L2 ball with the specified radius.
        """
        for param in self.nn.parameters():
            if param.requires_grad:
                param_norm = param.norm(p=2)
                if param_norm > self.l2_radius:
                    param.data = param * (self.l2_radius / param_norm)

    '''def clone(self):
        # Create a new agent with the same parameters and weights
        new_agent = Agent(
            q0=self.q0,
            K=self.K,
            in_size=self.in_size,
            out_size=self.out_size,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            l2_radius=self.l2_radius
        )
        # Detach and clone parameters
        new_state_dict = {key: val.detach().clone() for key, val in self.nn.state_dict().items()}
        new_agent.nn.load_state_dict(new_state_dict)
        return new_agent
    '''
    def clone(self):
        """
        Create a new agent with the same parameters, weights, and scheduler state.
        """
        # Create a new agent with the same initialization parameters
        new_agent = Agent(
            q0=self.q0,
            K=self.K,
            in_size=self.in_size,
            out_size=self.out_size,
            learning_rate=self.optimizer.param_groups[0]['lr'],  # Use current learning rate
            l2_radius=self.l2_radius
        )
        
        # Clone neural network parameters
        new_state_dict = {key: val.detach().clone() for key, val in self.nn.state_dict().items()}
        new_agent.nn.load_state_dict(new_state_dict)
        
        # Copy optimizer state
        new_agent.optimizer.load_state_dict(self.optimizer.state_dict())
        
        # Clone scheduler state
        new_agent.scheduler.last_epoch = self.scheduler.last_epoch  # Restore scheduler step count
        
        return new_agent


    