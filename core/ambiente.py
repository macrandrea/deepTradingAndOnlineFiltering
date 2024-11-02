# -*- coding: utf-8 -*-
"""
Created on Thu Nov 31 14:41:00 2024

@authors: macrandrea - gianlucapalmari
"""

import numpy as np

# Price Model simulating true price dynamics
class Price:
    def __init__(self, A, B, sigma_impact, sigma_price, N_assets):
        self.A = A
        self.B = B
        self.N_assets = N_assets
        self.sigma_impact = sigma_impact
        self.sigma_price = sigma_price

    def evolve_hidden_state(self, imp_tm1, u_tm1):
        noise = np.random.normal(0, self.sigma_impact, size=self.N_assets)
        return self.A @ imp_tm1 + self.B @ u_tm1 + noise

    def generate_price(self, S_tm1, theta, u_tm1, noise=True):
        noise_term = np.random.normal(0, self.sigma_price, size=self.N_assets) if noise else np.zeros(self.N_assets)
        price = S_tm1 + theta.reshape(self.N_assets, -1) @ u_tm1 + noise_term
        return price