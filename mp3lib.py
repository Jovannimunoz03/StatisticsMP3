#Functions for the main project in library that can be called on 
import numpy as np

#Calculates sample mean and unbiased covariance matrix
def sample_mean_cov(X):
    """Parameters:
        X: Data matrix with shape (N, d)
           N = number of samples
           d = number of random variables
    Returns:
        mean_vector: Sample mean vector
        cov_matrix: Sample covariance matrix"""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a 2-D array with at least two rows and one column.")
    #Calculate sample mean
    mean_vector = np.mean(X, axis=0)
    #Subtract the mean from every observation
    centered = X - mean_vector
    #Calculate unbiased sample covariance
    cov_matrix = (centered.T @ centered) / (X.shape[0] - 1)
    return mean_vector, cov_matrix


#Checks symmetry and positive semidefiniteness while checking matrix valid covariance matrix
def is_valid_covariance(S, tol=1e-10):
    """ A valid covariance matrix must:
        1. Be square.
        2. Be symmetric.
        3. Be positive semidefinite.
    Parameters:
        S: Matrix to check
        tol: Numerical tolerance
    Returns:
        True if valid, False otherwise."""

    S = np.asarray(S, dtype=float)
    #Check whether matrix is square
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return False
    if tol < 0:
        raise ValueError("tol must be nonnegative.")
    if not np.all(np.isfinite(S)):
        return False
    #Check symmetry
    if not np.allclose(S, S.T, atol=tol, rtol=0):
        return False
    #Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh((S + S.T) / 2)
    #Positive semidefinite means all eigenvalues >= 0
    return bool(np.min(eigenvalues) >= -tol)


#Calculates multivariate Gaussian probability density
def gaussian_pdf(x, mu, Sigma):
    """Parameters:
        x: Point or collection of points
           Shape (d,) or (N, d)
        mu: Mean vector
           Shape (d,)
        Sigma: Covariance matrix
               Shape (d, d)
    Returns:
        Gaussian probability density value(s)."""

    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    d = mu.size
    if d == 0 or Sigma.shape != (d, d):
        raise ValueError("Sigma must be square with dimensions matching mu.")
    #Determine whether x represents one point or multiple points
    one_point = x.ndim == 1
    if one_point:
        if x.shape != (d,):
            raise ValueError("A single x must have shape (d,).")
        x_rows = x.reshape(1, d)
    elif x.ndim == 2 and x.shape[1] == d:
        x_rows = x
    else:
        raise ValueError("x must have shape (d,) or (N, d).")
    #Covariance matrix must be symmetric
    if not np.allclose(Sigma, Sigma.T, atol=1e-10, rtol=0):
        raise ValueError("Sigma must be symmetric.")
    #Calculate determinant using log determinant
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Sigma must be positive definite.")
    #Difference between observations and mean
    delta = (x_rows - mu).T
    #Solve Sigma*z = delta instead of explicitly calculating inverse
    solved = np.linalg.solve(Sigma, delta)
    #Calculate squared Mahalanobis distance
    mahalanobis_squared = np.sum(delta * solved, axis=0)
    #Gaussian normalization constant in log form
    log_normalizer = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    #Calculate Gaussian PDF
    values = np.exp(log_normalizer - 0.5 * mahalanobis_squared)
    if one_point:
        return float(values[0])
    return values



#Creates matrix and removes correlation while calculating a whitening matrix W
def whitening_matrix(Sigma):
    """The whitening matrix satisfies:
        W @ Sigma @ W.T = I
    Parameters:
        Sigma: Positive-definite covariance matrix
    Returns:
        W: Whitening matrix"""

    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10, rtol=0):
        raise ValueError("Sigma must be symmetric.")
    #Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    #Whitening requires positive eigenvalues
    if np.any(eigenvalues <= 0):
        raise ValueError("Whitening requires a positive-definite covariance.")
    #Construct whitening matrix
    W = (np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T)
    return W


#Calculate optimal linear MMSE weights vector
def lmmse_weights(Sigma_xx, Sigma_sx):
    """The estimator is:
        s_hat = w.T @ x
    The weights satisfy:
        Sigma_xx @ w = Sigma_sx
    Parameters:
        Sigma_xx: Covariance matrix of observation vector x
        Sigma_sx: Cross-covariance vector between s and x
    Returns:
        w: LMMSE weight vector"""

    Sigma_xx = np.asarray(Sigma_xx, dtype=float)
    Sigma_sx = np.asarray(Sigma_sx, dtype=float).reshape(-1)
    if Sigma_xx.ndim != 2 or Sigma_xx.shape[0] != Sigma_xx.shape[1]:
        raise ValueError("Sigma_xx must be square.")
    if Sigma_sx.size != Sigma_xx.shape[0]:
        raise ValueError("Sigma_sx length must match Sigma_xx dimension.")
    if not np.allclose(Sigma_xx, Sigma_xx.T, atol=1e-10, rtol=0):
        raise ValueError("Sigma_xx must be symmetric.")
    #Solve the LMMSE normal equation
    w = np.linalg.solve(Sigma_xx, Sigma_sx)
    return w

# Bonus A
def covariance_factor(Sigma, method="cholesky", tol=1e-10):
    """Return A satisfying A @ A.T = Sigma, up to rounding."""
    S = np.asarray(Sigma, dtype=float)
    if (S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] == 0):
        raise ValueError("Sigma must be a nonempty square matrix.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative.")
    if not np.all(np.isfinite(S)):
        raise ValueError("Sigma must contain finite values.")
    if not np.allclose(S, S.T, atol=tol, rtol=0):
        raise ValueError("Sigma must be symmetric.")
    S = (S + S.T)/2
    if method == "cholesky":
        return np.linalg.cholesky(S)
    if method == "eigh":
        values, vectors = np.linalg.eigh(S)
        if values.min() < -tol:
            raise ValueError("Sigma must be positive semidefinite.")
        # Clip negative eigenvalues only within rounding tolerance.
        return vectors * np.sqrt(np.maximum(values, 0.0))
    raise ValueError("method must be 'cholesky' or 'eigh'.")

def pca_analysis(X, standardize=False):
    """PCA from the sample covariance, ordered by variance."""
    X = np.asarray(X, dtype=float)
    if (X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1):
        raise ValueError("X must have at least two rows and one column.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")
        
    mean, covariance = sample_mean_cov(X)
    
    if standardize:
        scale = np.sqrt(np.diag(covariance))
    else:
        scale = np.ones(X.shape[1])
    if np.any(scale <= 0):
        raise ValueError("Cannot standardize a constant column.")
    centered = (X - mean)/scale
    _, covariance = sample_mean_cov(centered)
    values, vectors = np.linalg.eigh(covariance)

    # PCA components are ordered from largest variance to smallest.
    order = np.argsort(values)[::-1]

    # The sample covariance is PSD; remove tiny negative roundoff.
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]

    if values.sum() <= 0:
        raise ValueError("PCA requires nonzero total variance.")

    return {
        "mean": mean,
        "scale": scale,
        "covariance": covariance,
        "eigenvalues": values,
        "eigenvectors": vectors,
        "explained": values/values.sum(),
        "scores": centered @ vectors
    }