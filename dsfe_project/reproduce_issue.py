
import numpy as np
import mne
from dsfe.fbcsp import FBCSP

def reproduce():
    # Simulate data: 48 trials, 3 channels, 250 samples
    n_trials = 48
    n_channels = 3
    n_times = 250
    fs = 250.0
    
    # Create random data
    X = np.random.randn(n_trials, n_channels, n_times)
    y = np.array([0, 1] * 24) # Balanced classes
    
    print(f"Data shape: {X.shape}")
    
    # Initialize FBCSP with 4 components (as in config)
    fbcsp = FBCSP(n_components=4)
    
    print("Attempting to fit FBCSP...")
    try:
        fbcsp.fit(X, y, fs)
        print("Fit successful (unexpected).")
    except Exception as e:
        print(f"Caught expected exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
