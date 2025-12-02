
import numpy as np
from mne.decoding import CSP

def reproduce():
    # Simulate data with 3 channels, similar to the user's log
    n_trials = 48
    n_channels = 3
    n_times = 250
    
    # Create random data
    # Make it slightly rank deficient or correlated to simulate real EEG issues
    X = np.random.randn(n_trials, n_channels, n_times)
    # Introduce correlation
    X[:, 2, :] = X[:, 0, :] + X[:, 1, :] * 0.1
    
    y = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
    
    print("Attempting CSP without regularization...")
    try:
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        csp.fit(X, y)
        print("Success without regularization (unexpected given the error).")
    except Exception as e:
        print(f"Caught expected error: {e}")

    print("\nAttempting CSP with regularization (ledoit_wolf)...")
    try:
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
        csp.fit(X, y)
        print("Success with regularization.")
    except Exception as e:
        print(f"Caught error with regularization: {e}")

    print("\nAttempting CSP with regularization (float)...")
    try:
        csp = CSP(n_components=4, reg=1e-4, log=True, norm_trace=False)
        csp.fit(X, y)
        print("Success with float regularization.")
    except Exception as e:
        print(f"Caught error with float regularization: {e}")

if __name__ == "__main__":
    reproduce()
