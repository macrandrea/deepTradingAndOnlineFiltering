import torch
import numpy as np
from scipy.optimize import minimize

class KalmanFilter:
    def __init__(self, M, A_init=None, B_init=None, Q=None, R=None):
        """
        Initialize the Kalman Filter with dimensions and noise covariances.

        Parameters:
        - M: Dimension parameter for control and observation vectors.
        - Q: Process noise covariance matrix (torch.Tensor, shape: (M^2, M^2)).
        - R: Measurement noise covariance matrix (torch.Tensor, shape: (M, M)).
        """
        self.M = M
        self.state_dim = M ** 2
        self.control_dim = M
        self.obs_dim = M

        # Initialize state and covariance with torch tensors
        self.x = torch.abs(torch.randn(self.state_dim))  # Initial state
        self.P = torch.eye(self.state_dim)    # Initial covariance
        self.Q = Q if Q is not None else torch.eye(self.state_dim)  # Process noise
        self.R = R if R is not None else torch.eye(self.obs_dim)    # Measurement noise

        # Initialize matrices A and B for MLE estimation
        self.A = A_init if A_init is not None else torch.randn(self.state_dim, self.state_dim)
        self.B = B_init if B_init is not None else torch.randn(self.state_dim, self.control_dim)

    def reset(self, initial_state=None, initial_covariance=None):
        """Reset the filter's state and covariance to specified initial values."""
        self.x = initial_state if initial_state is not None else torch.abs(torch.randn(self.state_dim))
        self.P = initial_covariance if initial_covariance is not None else torch.eye(self.state_dim)

    def compute_C(self):
        """Reshape the current state vector x into a (M, M) observation matrix."""
        return self.x.view(self.M, self.M)

    def predict(self, u_k):
        """Predict the next state and covariance based on control input."""

        x_pred = self.A.float() @ self.x + self.B.float() @ u_k
        P_pred = self.A.float() @ self.P.float() @ self.A.mT.float() + self.Q.float()
        return x_pred, P_pred
    
    def update(self, y_k, x_pred, P_pred, u_k):
        """Update state and covariance with the current observation."""
        H = torch.kron(u_k.unsqueeze(0), torch.eye(self.M, device=u_k.device))  # Observation matrix from control
        y_pred = H @ x_pred  # Observation prediction (without reshaping x_pred)
        S = H @ P_pred @ H.T + self.R  # Innovation covariance
        K = P_pred @ H.T @ torch.linalg.inv(S)  # Kalman gain
        # Update state and covariance
        self.x = x_pred + K @ (y_k - y_pred)
        self.P = (torch.eye(self.state_dim, device=u_k.device) - K @ H) @ P_pred
        return self.x

    def fit(self, Y, U):
        """Fit the A and B matrices by maximizing the likelihood of observations."""
        def log_likelihood(params, Y, U):
            # Convert parameters to torch tensors without requiring gradients
            A = torch.tensor(params[:self.state_dim**2].reshape(self.state_dim, self.state_dim), requires_grad=False)
            B = torch.tensor(params[self.state_dim**2:].reshape(self.state_dim, self.control_dim), requires_grad=False)
            self.reset()  # Reset state and covariance for the fitting

            log_likelihood = 0.0
            Y_torched = torch.tensor(Y)
            U_torched = torch.tensor(U)
            for t in range(Y_torched.shape[1]):
                y_k = Y_torched[t]
                u_k = U_torched[t]

                # Prediction step
                x_pred, P_pred = self._predict_step(A, B, u_k)
                H = torch.kron(u_k.unsqueeze(0), torch.eye(self.M, device=u_k.device))
                y_pred = H @ x_pred  # Avoid reshaping x_pred here
                residual = y_k - y_pred

                # Innovation covariance
                S = H @ P_pred @ H.T + self.R
                log_det_S = torch.linalg.slogdet(S)[1]  # More stable log-det calculation
                log_likelihood -= 0.5 * (residual.T @ torch.linalg.inv(S) @ residual + log_det_S)

                # Update step
                self._update_step(H, residual, x_pred, P_pred, S)
            return -log_likelihood.item()  # Convert to a scalar for scipy.optimize
        # Flatten initial parameters as a numpy array for scipy.optimize
        initial_params = np.hstack([self.A.numpy().ravel(), self.B.numpy().ravel()])
        result = minimize(log_likelihood, initial_params, args=(Y, U), method='L-BFGS-B')

        # Update A and B matrices with optimized values
        opt_params = result.x
        self.A = torch.tensor(opt_params[:self.state_dim**2].reshape(self.state_dim, self.state_dim)  )
        self.B = torch.tensor(opt_params[self.state_dim**2:].reshape(self.state_dim, self.control_dim))

    def _predict_step(self, A, B, u_k):
        """Helper for internal predict step during likelihood calculation."""
        x_pred = A.float() @ self.x + B.float() @ u_k
        P_pred = A.float() @ self.P.float() @ A.T.float() + self.Q.float()
        return x_pred, P_pred

    def _update_step(self, H, residual, x_pred, P_pred, S):
        """Helper for internal update step during likelihood calculation."""
        K = P_pred @ H.T @ torch.linalg.inv(S)
        self.x = x_pred + K @ residual
        self.P = (torch.eye(self.state_dim) - K @ H) @ P_pred

class DualKalmanFilterSystem:
    def __init__(self, M, A1_init=None, B1_init=None, A2_init=None, B2_init=None, Q1=None, R1=None, Q2=None, R2=None):
        """Initialize dual Kalman filters for two separate models."""
        self.kf1 = KalmanFilter(M, A_init=A1_init, B_init=B1_init, Q=Q1, R=R1)
        self.kf2 = KalmanFilter(M, A_init=A2_init, B_init=B2_init, Q=Q2, R=R2)

    def predict(self, u_k):
        """Predict the next states for both filters."""
        x1_pred, P1_pred = self.kf1.predict(u_k)
        x2_pred, P2_pred = self.kf2.predict(u_k)
        return (x1_pred, P1_pred), (x2_pred, P2_pred)

    def update(self, y_k1, y_k2, x1_pred, P1_pred, x2_pred, P2_pred):
        """Update the states of both filters based on their respective observations."""
        x1_updated = self.kf1.update(y_k1, x1_pred, P1_pred)
        x2_updated = self.kf2.update(y_k2, x2_pred, P2_pred)
        return x1_updated, x2_updated

    def filter_step(self, y_k1, y_k2, u_k):
        """Executes a filtering step for both filters with their respective observations and control input."""
        (x1_pred, P1_pred), (x2_pred, P2_pred) = self.predict(u_k)
        x1_filtered = self.kf1.update(y_k1, x1_pred, P1_pred)
        x2_filtered = self.kf2.update(y_k2, x2_pred, P2_pred)
        return x1_filtered, x2_filtered
