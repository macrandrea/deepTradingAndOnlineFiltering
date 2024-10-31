import ambiente as amb
import numpy as np

class KalmanFilter():
    def __init__(self, A, B, C, D, Q, R, P, x0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def update(self, u, z):
        # Prediction update
        x_hat = self.A @ self.x + self.B @ u
        P_hat = self.A @ self.P @ self.A.T + self.Q

        # Measurement update
        K = P_hat @ self.C.T @ np.linalg.inv(self.C @ P_hat @ self.C.T + self.R)
        self.x = x_hat + K @ (z - self.C @ x_hat)
        self.P = P_hat - K @ self.C @ P_hat

        return self.x
    
    def get_state(self):
        return self.x
    


class Agente():
    def __init__(self):
        self.ambiente = amb.Price()
