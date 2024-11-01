import numpy as np
from scipy.optimize import minimize

class KalmanFilterWithMLE:
    def __init__(self, M, Q, R):
        """
        Initializes the Kalman Filter with dimensions and noise covariances.

        Parameters:
        - M: Dimension parameter for control and observation vectors.
        - Q: Process noise covariance matrix (shape: (2*M^2, 2*M^2)).
        - R: Measurement noise covariance matrix (shape: (M, M)).
        """
        # Model dimensions
        self.M = M
        self.state_dim = 2 * M**2  # Dimension of the state vector x
        self.control_dim = M       # Dimension of the control vector u
        self.obs_dim = M           # Dimension of the observation vector y

        # Initial state and covariance
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)
        self.Q = Q                 # Process noise covariance
        self.R = R                 # Measurement noise covariance

        # Initialize matrices A, B, C with random values for MLE estimation
        self.A = np.random.randn(self.state_dim, self.state_dim)
        self.B = np.random.randn(self.state_dim, self.control_dim)

    def reset(self, initial_state=None, initial_covariance=None):
        """
        Resets the filter's state and covariance to specified initial values.

        Parameters:
        - initial_state: Optional, initial state vector (default: zero vector).
        - initial_covariance: Optional, initial state covariance matrix (default: identity matrix).
        """
        self.x = initial_state if initial_state is not None else np.zeros(self.state_dim)
        self.P = initial_covariance if initial_covariance is not None else np.eye(self.state_dim)

    def predict_S_tilde(self, u_k):
        """
        Predicts the next state and covariance based on the current state and control input.

        Parameters:
        - u_k: Current control input (shape: (control_dim,))

        Returns:
        - x_pred: Predicted state (shape: (state_dim,))
        - P_pred: Predicted covariance (shape: (state_dim, state_dim))
        """
        x_eta_t_tm1 = self.A[1] @ self.x[2*self.M**2:] + self.B[1] @ u_k
        return x_eta_t_tm1
    
    def predict_S(self, u_k):
        """
        Predicts the next state and covariance based on the current state and control input.

        Parameters:
        - u_k: Current control input (shape: (control_dim,))

        Returns:
        - x_pred: Predicted state (shape: (state_dim,))
        - P_pred: Predicted covariance (shape: (state_dim, state_dim))
        """
        x_theta_t_tm1 = self.A[0] @ self.x[:2*self.M**2] + self.B[0] @ u_k
        return x_theta_t_tm1
    
    def predict(self, u_k):
        """
        Predicts the next state and covariance based on the current state and control input.

        Parameters:
        - u_k: Current control input (shape: (control_dim,))

        Returns:
        - x_pred: Predicted state (shape: (state_dim,))
        - P_pred: Predicted covariance (shape: (state_dim, state_dim))
        """
        x_pred = self.A @ self.x + self.B @ u_k
        P_pred = self.A @ self.P @ self.A.T + self.Q
        return x_pred, P_pred

    def update(self, y_k, x_pred, P_pred):
        """
        Updates the state and covariance based on the observation.

        Parameters:
        - y_k: Observation at the current time step (shape: (obs_dim,))
        - x_pred: Predicted state from the predict step (shape: (state_dim,))
        - P_pred: Predicted covariance from the predict step (shape: (state_dim, state_dim))

        Returns:
        - x_updated: Updated (filtered) state vector after incorporating observation.
        """
        # Define C
        
        # Observation prediction
        y_pred = C @ x_pred
        # Innovation covariance
        S = C @ P_pred @ C.T + self.R
        # Kalman gain
        K = P_pred @ C.T @ np.linalg.inv(S)

        # State update with Kalman gain
        self.x = x_pred + K @ (y_k - y_pred)
        self.P = (np.eye(self.state_dim) - K @ C) @ P_pred
        return self.x

    def log_likelihood(self, params, Y, U):
        """
        Computes the negative log-likelihood of the observed data given the parameters.

        Parameters:
        - params: Flattened array of parameters A, B, C for optimization.
        - Y: Sequence of observations (shape: (obs_dim, time_steps))
        - U: Sequence of control inputs (shape: (control_dim, time_steps))

        Returns:
        - Negative log-likelihood of the observations given the parameters.
        """
        # Unpack parameters from flat array
        A = params[:self.state_dim**2].reshape(self.state_dim, self.state_dim)
        B = params[self.state_dim**2:self.state_dim**2 + self.state_dim * self.control_dim].reshape(self.state_dim, self.control_dim)
        
        # Initialize/reset state and covariance
        self.reset()

        log_likelihood = 0.0
        # Iterate over the time steps
        for t in range(Y.shape[1]):
            y_k = Y[:, t]
            u_k = U[:, t]
            
            # Predict step
            x_pred, P_pred = self._predict_step(A, B, u_k)
            C = 
            # Calculate observation residual
            y_pred = C @ x_pred
            residual = y_k - y_pred
            # Innovation covariance
            S = C @ P_pred @ C.T + self.R
            # Log-likelihood contribution for this step
            log_likelihood += -0.5 * (residual.T @ np.linalg.inv(S) @ residual + np.log(np.linalg.det(S)))
            
            # Update step
            self._update_step(C, residual, x_pred, P_pred, S)

        return -log_likelihood

    def fit(self, Y, U):
        """
        Fits the parameters A, B, and C by maximizing the likelihood of the observed data.

        Parameters:
        - Y: Observed data sequence (shape: (obs_dim, time_steps))
        - U: Control input sequence (shape: (control_dim, time_steps))

        Returns:
        - Optimized matrices A, B, C after fitting.
        """
        # Initial parameter vector as a flattened array of A, B, C
        initial_params = np.hstack([self.A.ravel(), self.B.ravel(), self.C.ravel()])
        
        # Minimize the negative log-likelihood using scipy's minimize function
        result = minimize(self.log_likelihood, initial_params, args=(Y, U), method='L-BFGS-B')
        
        # Reshape optimized parameters back to matrices
        opt_params = result.x
        self.A = opt_params[:self.state_dim**2].reshape(self.state_dim, self.state_dim)
        self.B = opt_params[self.state_dim**2:self.state_dim**2 + self.state_dim * self.control_dim].reshape(self.state_dim, self.control_dim)
        self.C = opt_params[self.state_dim**2 + self.state_dim * self.control_dim:].reshape(self.obs_dim, self.state_dim)

        return self.A, self.B, self.C

    def filter_step(self, y_k, u_k):
        """
        Executes a single filtering step with the current observation and control input.

        Parameters:
        - y_k: Current observation (shape: (obs_dim,))
        - u_k: Current control input (shape: (control_dim,))

        Returns:
        - x_filtered: Updated state estimate.
        - state_residual: Residual (difference between filtered and predicted state).
        """
        # Predict state and covariance
        x_pred, P_pred = self.predict(u_k)
        
        # Update state based on current observation
        x_filtered = self.update(y_k, x_pred, P_pred)
        
        # Compute residual (difference between filtered and predicted state)
        state_residual = x_filtered - x_pred
        
        return x_filtered, state_residual
    
    def _predict_step(self, A, B, u_k):
        """Helper function for internal predict step during likelihood calculation."""
        x_pred = A @ self.x + B @ u_k
        P_pred = A @ self.P @ A.T + self.Q
        return x_pred, P_pred

    def _update_step(self, C, residual, x_pred, P_pred, S):
        """Helper function for internal update step during likelihood calculation."""
        K = P_pred @ C.T @ np.linalg.inv(S)
        self.x = x_pred + K @ residual
        self.P = (np.eye(self.state_dim) - K @ C) @ P_pred
