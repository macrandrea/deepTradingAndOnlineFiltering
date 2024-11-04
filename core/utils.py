import numpy as np 


def generate_matrix_with_positive_real_eigenvalues(size, eigenvalues=None):
    """
    Generates a matrix X_0 with specified real positive eigenvalues.
    
    Parameters:
    - size: int
        The size of the matrix X_0 (size x size).
    - eigenvalues: list or numpy.ndarray, optional
        Specific positive eigenvalues for the matrix. If None, random positive values will be generated.
    
    Returns:
    - X_0: numpy.ndarray
        A matrix X_0 with the specified positive real eigenvalues.
    """
    if eigenvalues is None:
        # Generate random positive eigenvalues if none are specified
        eigenvalues = np.random.uniform(0.5, 2.0, size)
    else:
        # Ensure provided eigenvalues are positive and match the specified size
        assert len(eigenvalues) == size, "Number of eigenvalues must match the matrix size"
        assert np.all(np.array(eigenvalues) > 0), "Eigenvalues must be positive"

    # Create a diagonal matrix with the specified eigenvalues
    D = np.diag(eigenvalues)

    # Generate a random invertible matrix Q (not necessarily orthogonal)
    Q = np.random.randn(size, size)
    while np.linalg.det(Q) == 0:
        Q = np.random.randn(size, size)  # Ensure Q is invertible

    # Construct X_0 = Q D Q^(-1), which will have the specified eigenvalues but may not be symmetric
    X_0 = Q @ D @ np.linalg.inv(Q)
    return X_0

def construct_decay_matrix(X_0, decay_type='homogeneous', decay_rate=0.9, eigenvalues=None):
    """
    Constructs a matrix A for evolving X(t+1) = A X(t) such that X(t) maintains positive
    real eigenvalues that decrease over time.

    Parameters:
    - X_0: numpy.ndarray
        Initial matrix X_0 with real, positive eigenvalues.
    - decay_type: str, optional
        Type of decay for the eigenvalues of A. Options are:
        'homogeneous' - all eigenvalues decay at the same rate (default decay_rate).
        'heterogeneous' - decay rates are randomly sampled in (0, 0.99) if eigenvalues not specified.
    - decay_rate: float, optional
        The decay factor applied to all eigenvalues in the homogeneous case (default is 0.9).
    - eigenvalues: list or numpy.ndarray, optional
        Custom decay factors for heterogeneous decay, one for each eigenvalue of X_0.

    Returns:
    - A: numpy.ndarray
        The decay matrix A.
    """
    # Ensure X_0 is square
    assert X_0.shape[0] == X_0.shape[1], "X_0 must be a square matrix"
    
    # Diagonalize X_0
    eigvals, eigvecs = np.linalg.eig(X_0)
    
    # Ensure eigenvalues of X_0 are real and positive
    if not np.all(eigvals > 0):
        raise ValueError("X_0 must have positive real eigenvalues")
    
    # Construct the diagonal decay matrix Lambda based on decay type
    if decay_type == 'homogeneous':
        # Homogeneous decay: same decay rate for all eigenvalues
        Lambda = np.diag([decay_rate] * len(eigvals))
    elif decay_type == 'heterogeneous':
        # Heterogeneous decay: random decay rates in (0, 0.99) if eigenvalues not specified
        if eigenvalues is None:
            eigenvalues = np.random.uniform(0.01, 0.99, len(eigvals))
        elif len(eigenvalues) != len(eigvals):
            raise ValueError("For heterogeneous decay, provide a decay rate for each eigenvalue.")
        Lambda = np.diag(eigenvalues)
    else:
        raise ValueError("Invalid decay_type. Choose 'homogeneous' or 'heterogeneous'.")
    
    # Construct the decay matrix A using the same eigenbasis as X_0
    A = eigvecs @ Lambda @ np.linalg.inv(eigvecs)
    
    return A
