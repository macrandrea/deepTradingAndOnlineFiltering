import core.ambiente as amb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define neural network for agent control
class DNNet(nn.Module):
    def __init__(self, in_size, hidden_size=30, num_layers=5):
        super(DNNet, self).__init__()
        layers = [nn.Linear(in_size, hidden_size), nn.LeakyReLU()]
        layers += [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Agent Class with control and training functionalities
class Agent:
    def __init__(self, K, learning_rate=0.001):
        self.nn = DNNet(in_size=2)
        self.optimizer = optim.AdamW(self.nn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.K = K
        self.step_count = 0

    def generate_control(self, residuals_fundamental, residuals_efficient):
        self.step_count += 1
        input_data = torch.tensor([residuals_fundamental, residuals_efficient], dtype=torch.float32)
        return self.nn(input_data)

    def train(self, loss_elements):
        if self.step_count >= self.K:
            loss = torch.mean(torch.square(loss_elements))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def copy(self):
        """Creates a new agent with the same neural network weights."""
        new_agent = Agent(K=self.K, learning_rate=self.optimizer.param_groups[0]['lr'])
        new_agent.nn.load_state_dict(self.nn.state_dict())  # Copy the neural network weights
        return new_agent