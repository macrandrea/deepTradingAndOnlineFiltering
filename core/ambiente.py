# -*- coding: utf-8 -*-
"""
Created on Thu Nov 31 14:41:00 2024

@authors: macrandrea - gianlucapalmari
"""

import torch
import copy

class PriceModel:
    def __init__(self, M, A1_init=None, B1_init=None, A2_init=None, B2_init=None,
                 sigma_x_1=None, sigma_x_2=None, sigma_y_1=None, sigma_y_2=None):
        # Dimensions
        self.M = M # Num assets 
        self.M2 = M ** 2 

        # Fixed matrices for each price model
        self.A1 = A1_init if A1_init is not None else torch.rand((self.M2, self.M2))     # State transition matrix for x1
        self.A2 = A2_init if A2_init is not None else torch.rand((self.M2, self.M2))     # State transition matrix for x2
        self.B1 = B1_init if B1_init is not None else torch.rand((self.M2, M))           # Control matrix for x1
        self.B2 = B2_init if B2_init is not None else torch.rand((self.M2, M))           # Control matrix for x2

        # Initializing state vectors (flattened matrix representation)
        self.x1_k = 1.e-3 * torch.ones(self.M2)             # State vector for price model 1
        self.x2_k = 1.e-3 * torch.ones(self.M2)             # State vector for price model 2

        # Control input 
        self.u_k = torch.zeros(M)                    # Control input vector
        
        # Measurement noise for each price model
        self.v_k1 = torch.zeros(M)                   # Measurement noise for model 1
        self.v_k2 = torch.zeros(M)                   # Measurement noise for model 2

        # Noise covariance matrices
        self.Q_1 = sigma_x_1 * torch.eye(self.M2) if sigma_x_1 is not None else  1.e-4 * torch.eye(self.M2) # Process noise covariance for x1
        self.Q_2 = sigma_x_2 * torch.eye(self.M2) if sigma_x_2 is not None else  1.e-4 * torch.eye(self.M2) # Process noise covariance for x2
        self.R_1 = sigma_y_1 * torch.eye(self.M) if sigma_x_1 is not None else  1.e-2 * torch.eye(self.M)   # Measurement noise covariance for model 1
        self.R_2 = sigma_y_2 * torch.eye(self.M) if sigma_x_1 is not None else  1.e-2 * torch.eye(self.M) # Measurement noise covariance for model 2

    def update_states(self):
        """
        Prediction step for each state vector.
        `x1` and `x2` are updated independently using A1, A2, B1, and B2.
        """

        # noise generation
        noise_x_1 = self.Q_1 @ torch.randn(self.M2)
        noise_x_2 = self.Q_2 @ torch.randn(self.M2)

        # Update x1 using A1, B1, control input, and process noise
        self.x1_k = self.A1 @ self.x1_k + self.B1 @ torch.abs(self.u_k) + noise_x_1
        
        # Update x2 using A2, B2, control input, and process noise
        self.x2_k = self.A2 @ self.x2_k + self.B2 @ torch.abs(self.u_k) + noise_x_2
    
    def update_returns(self):
        """
        Update_returns y1 and y2 based on current states x1 and x2.
        Reshapes `x1` as `C1` and uses `C2` separately.
        """

        # noise generation
        noise_y_1 = self.R_1 @ torch.randn(self.M)
        noise_y_2 = self.R_2 @ torch.randn(self.M)

        # Create output matrices based on current states
        C1 = self.x1_k.view(self.M, self.M)  # Reshape x1_k as output matrix for y1
        C2 = self.x2_k.view(self.M, self.M)  # Reshape x2_k as output matrix for y2
        
        # Output y1 and y2 based on current states and measurement noise
        y1 = C1 @ self.u_k + noise_y_1
        y2 = C2 @ self.u_k + noise_y_2

        return y1, y2

    def update_control(self, new_control_input):
        """
        Updates the control input for the next time step.
        """
        with torch.no_grad():
            self.u_k = new_control_input
            
    def copy(self):
        """
        Create a deep copy of the current PriceModel instance.
        Returns:
            A new PriceModel instance with identical values, independent of the original.
        """
        return copy.deepcopy(self)