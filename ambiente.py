# -*- coding: utf-8 -*-
"""
Created on Thu Nov 31 14:41:00 2024

@authors: macrandrea - gianlucapalmari
"""

import numpy as np

class Price():

    def __init__(self, price_0, imp_0, A_0, B_0, C_0, D_0, y_0, u_t, N_assets):
        self.price = price_0
        self.imp = imp_0
        self.A = A_0
        self.B = B_0
        self.C = C_0
        self.D = D_0    
        self.y = y_0
        self.u_t = u_t
        self.N_assets = N_assets

    def evolve_hidden(self, imp_t, A_t, B_t, u_t, N_assets):
        '''
        returns: hidden state at time t+1
        pmt
        imp_t: hidden state at time t
        A_t: matrix A at time t
        B_t: matrix B at time t
        u_t: control at time t
        N_assets: number of assets
        ''' 

        w_t = np.random.normal(0, 0.1, size=N_assets)

        self.imp = A_t @ imp_t + B_t @ u_t + w_t

        return self.imp

    def set_price_ret(self, imp_t, C_t, D_t, u_t, N_assets, noise = True):
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
            v_t = np.random.normal(0, 0.1, size=N_assets)
        else:
            v_t = np.zeros(N_assets)

        price_ret_t = C_t @ imp_t + D_t @ u_t + v_t

        return price_ret_t

    def get_price_ret(self, u_t):
        '''
        returns: price_ret at time t+1
        pmt:
        u_t: control at time t
        '''

        impact = self.evolve_hidden(self.imp, self.A, self.B, u_t, self.N_assets)

        price = self.set_price_ret(impact, self.C, self.D, u_t, self.N_assets)

        return price , impact     