# -*- coding: utf-8 -*-
"""
Created on Thu Nov 31 14:41:00 2024

@authors: macrandrea - gianlucapalmari
"""

import numpy as np

class Price():

    def __init__(self, A_0, B_0, sigma_noise_impact, 
                 sigma_noise_price, N_assets):
        self.A = A_0
        self.B = B_0
        self.N_assets = N_assets
        self.sigma_noise_impact = sigma_noise_impact
        self.sigma_noise_price = sigma_noise_price

    def evolve_hidden(self, imp_tm1, u_tm1, N_assets):
        '''
        returns: hidden state at time t+1
        pmt
        imp_t: hidden state at time t
        A_t: matrix A at time t
        B_t: matrix B at time t
        u_t: control at time t
        N_assets: number of assets
        ''' 

        w_t = np.random.normal(0, self.sigma_noise_impact, size=N_assets)

        imp_t = self.A @ imp_tm1 + self.B @ u_tm1 + w_t

        return imp_t
    
    def fundamental_price(self, S_tm1, theta_t, u_tm1, N_assets, noise = False):
        '''
        returns: price_ret at time t+1
        pmt:
        price_t: price_ret at time t
        imp_t: hidden state at time t
        C_t: matrix C at time t
        D_t: matrix D at time t
        u_t: control at time t
        '''
        if noise:
            v_t = np.random.normal(0, self.sigma_noise_price, size=N_assets)
        else:
            v_t = np.zeros(N_assets)

        S_t = S_tm1 + theta_t.reshape(self.N_assets,-1) @ u_tm1 + v_t
        
        return S_t
    
    def efficient_price(self, S_tm1, eta_t, u_t, noise = False):
        '''
        returns: price_ret at time t+1
        pmt:
        price_t: price_ret at time t
        imp_t: hidden state at time t
        C_t: matrix C at time t
        D_t: matrix D at time t
        u_t: control at time t
        '''
        if noise:
            v_t = np.random.normal(0, 0.1, size=self.N_assets)
        else:
            v_t = np.zeros(self.N_assets)

        S_tilde_t = S_tm1 + eta_t.reshape(self.N_assets,-1) @ u_t + v_t

        return S_tilde_t
        
    