
import numpy as np
from scipy.linalg import fractional_matrix_power, logm
from . import config

# Try importing pyriemann, but provide fallback or error if missing
try:
    from pyriemann.utils.covariance import covariances
    from pyriemann.utils.mean import mean_riemann
except ImportError:
    print("Warning: pyriemann not found. RG features will fail.")

def compute_fta_features(X, fs, band):
    """
    Compute Fourier Transform Amplitude features.
    X: (n_trials, n_channels, n_times)
    band: tuple (f1, f2)
    """
    f1, f2 = band
    n_trials, n_channels, n_times = X.shape
    
    # FFT frequencies
    freqs = np.fft.rfftfreq(n_times, d=1.0/fs)
    
    # Indices for the band
    idx_band = np.where((freqs >= f1) & (freqs <= f2))[0]
    
    if len(idx_band) == 0:
        # Handle case where band is empty (e.g. very narrow band or wrong fs)
        return np.zeros((n_trials, n_channels)) # Return something or raise error
        
    F_list = []
    for trial in range(n_trials):
        # FFT along time axis
        fft_vals = np.fft.rfft(X[trial], axis=-1)
        # Amplitude
        amp = np.abs(fft_vals)[:, idx_band]
        # Flatten: concatenate amplitudes of all channels
        F_list.append(amp.reshape(-1))
        
    F = np.stack(F_list, axis=0)
    return F

def compute_rg_features(X, fs, band=None):
    """
    Compute Riemannian Geometry features.
    X: (n_trials, n_channels, n_times)
    band: tuple (f1, f2) or None. If provided, X is assumed to be NOT filtered yet?
          Actually, usually we filter X before calling this. 
          But for FDCC, we might pass a band.
          Let's assume X is ALREADY filtered to the band if band is None.
          If band is provided, we should filter it? 
          The prompt's example for FDCC says: "extract_feature_for_band... if rg: bandpass_X... compute_rg".
          So let's assume X passed here is already the signal we want to compute Cov on.
    """
    # Calculate Covariance Matrices
    # pyriemann covariances expects (n_matrices, n_channels, n_times)
    covs = covariances(X, estimator='oas') # 'oas' or 'scm' (sample covariance)
    
    # Riemannian Mean (Geometric Mean)
    # This should be computed on Training data and applied to Test data?
    # The paper says: "Use Riemannian mean of all training samples P_G"
    # Then map to tangent space.
    # So we need to return the features.
    # But wait, if we are in training, we compute P_G from X.
    # If we are in testing, we need P_G from training!
    # This function computes features for a batch X.
    # If we want to support Train/Test split properly, we need a class or pass P_G.
    # For simplicity in this functional design, let's assume this is for a batch and we compute P_G from it.
    # BUT strictly speaking, for Test data, we should use Train P_G.
    # The prompt's example `compute_rg_features` calculates P_G inside. 
    # This implies it's "Training mode" or "Unsupervised/Transductive"?
    # Let's stick to the prompt's simplicity for now, but add a TODO.
    
    P_G = mean_riemann(covs)
    
    # Map to Tangent Space
    P_G_inv_sqrt = fractional_matrix_power(P_G, -0.5)
    
    feats = []
    for P in covs:
        # Map to tangent space
        T = P_G_inv_sqrt @ P @ P_G_inv_sqrt
        L = logm(T)
        
        # Vectorize (Upper triangle)
        # Multiply off-diagonals by sqrt(2)
        upper = L[np.triu_indices_from(L)]
        
        # Correct scaling for off-diagonals to preserve norm
        # The prompt says: "diagonal 1, off-diagonal sqrt(2)"?
        # Actually standard is: off-diagonals * sqrt(2).
        # Let's implement the scaling.
        
        # We need to identify which are off-diagonals in the flattened vector.
        # It's easier to do it on the matrix L before extracting upper.
        
        # Create a mask for off-diagonals
        mask = np.ones_like(L, dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Apply sqrt(2) to off-diagonals
        L_scaled = L.copy()
        L_scaled[mask] *= np.sqrt(2) # Wait, if we do this, we double count?
        # No, we only take upper triangle later.
        # But we need to be careful. The metric is ||A||_F.
        # The vector norm should equal the matrix norm.
        # ||L||_F^2 = sum(diag^2) + 2*sum(upper_off^2).
        # Vector v = [diag, sqrt(2)*upper_off].
        # ||v||^2 = sum(diag^2) + sum(2 * upper_off^2) = ||L||_F^2.
        # So yes, multiply off-diagonals by sqrt(2).
        
        # However, if I multiply the WHOLE off-diagonal by sqrt(2), then take upper, it works.
        # But wait, `logm` returns a symmetric matrix for SPD inputs.
        
        # Let's follow the prompt's snippet logic exactly to be safe.
        # "upper[upper_mask[np.triu_indices_from(L)]] *= np.sqrt(2)"
        
        upper = L[np.triu_indices_from(L)]
        # We need to find indices in 'upper' that correspond to off-diagonals.
        # The prompt's code:
        # diag_idx = np.diag_indices_from(L)
        # mask_offdiag = np.ones_like(L, dtype=bool)
        # mask_offdiag[diag_idx] = False
        # upper_mask = np.triu(mask_offdiag) # Upper triangle of off-diagonal mask
        # upper[upper_mask[np.triu_indices_from(L)]] *= np.sqrt(2)
        
        # This logic in the prompt seems slightly complex to map 'upper_mask' to 'upper' indices.
        # Let's simplify:
        # The 'upper' vector contains elements in row-major order (or whatever triu_indices returns).
        # We can just iterate or construct the vector manually.
        
        n = L.shape[0]
        vec = []
        for i in range(n):
            for j in range(i, n):
                val = L[i, j]
                if i != j:
                    val *= np.sqrt(2)
                vec.append(val)
        feats.append(np.array(vec))
        
    G = np.stack(feats, axis=0)
    return G, P_G # Return P_G so we can reuse it if needed (though prompt didn't ask)

def compute_rg_features_with_mean(X, fs, P_G):
    """
    Compute RG features using a pre-computed Riemannian Mean P_G.
    """
    covs = covariances(X, estimator='oas')
    P_G_inv_sqrt = fractional_matrix_power(P_G, -0.5)
    
    feats = []
    for P in covs:
        T = P_G_inv_sqrt @ P @ P_G_inv_sqrt
        L = logm(T)
        n = L.shape[0]
        vec = []
        for i in range(n):
            for j in range(i, n):
                val = L[i, j]
                if i != j:
                    val *= np.sqrt(2)
                vec.append(val)
        feats.append(np.array(vec))
    return np.stack(feats, axis=0)
