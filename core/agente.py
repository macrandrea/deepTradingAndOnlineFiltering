import core.ambiente as amb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define neural network for agent control
class DNNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=2, num_layers=2):
        super(DNNet, self).__init__()
        
        # Initialize the layers with LeakyReLU activations for the hidden layers
        layers = [nn.Linear(in_size, hidden_size), nn.LeakyReLU()]
        
        # Add additional hidden layers with LeakyReLU
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        
        # Add the output layer with Tanh activation
        layers.append(nn.Linear(hidden_size, out_size))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Agent Class with control and training functionalities
class Agent:
    def __init__(self, K, in_size, out_size, learning_rate=0.001):
        self.nn = DNNet(in_size=in_size, out_size=out_size)
        self.optimizer = optim.AdamW(self.nn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.K = K
        self.step_count = 0
        self.l2_radius = 1.

    def generate_control(self, residuals_fundamental, residuals_efficient):
        self.step_count += 1
        print(residuals_fundamental)
        input_data = torch.cat([residuals_fundamental.detach(), residuals_efficient.detach()], dim=0).detach()
        return self.nn(input_data)

    def train(self, loss):
        if self.step_count >= self.K:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Project weights onto L2 ball
            self.project_weights_to_l2_ball()

    def project_weights_to_l2_ball(self):
        """Projects all network parameters onto an L2 ball with the specified radius."""
        for param in self.nn.parameters():
            if param.requires_grad:
                # Compute the L2 norm of the parameter tensor
                param_norm = param.norm(p=2)
                if param_norm > self.l2_radius:
                    # Scale the parameters to lie within the L2 ball
                    param.data = param * (self.l2_radius / param_norm)

        #for param in self.nn.parameters():
        #    print(param)

    def copy(self):
        """Creates a new agent with the same neural network weights."""
        new_agent = Agent(K=self.K, learning_rate=self.optimizer.param_groups[0]['lr'])
        new_agent.nn.load_state_dict(self.nn.state_dict())  # Copy the neural network weights
        return new_agent
    