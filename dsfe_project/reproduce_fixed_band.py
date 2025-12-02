
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dsfe import config, train_eval, fdcc

# Mock Data
X = np.random.rand(20, 22, 100) # 20 trials, 22 channels, 100 samples
y = np.array([0, 1] * 10)
fs = 250.0

# Configure for test
config.FIXED_BEST_BAND_MODE = True
config.USE_FDCC = True
config.USE_FTA = True
config.USE_RG = True
config.N_FOLDS_EVAL = 2 # 2 folds for speed
config.VERBOSE = True

# Mock fdcc_select_band to track calls
original_fdcc_select_band = fdcc.fdcc_select_band
fdcc.fdcc_select_band = MagicMock(side_effect=lambda X, y, fs, ftype: (8, 12) if ftype == 'fta' else (10, 14))

print("Running evaluate_session with FIXED_BEST_BAND_MODE=True...")
train_eval.evaluate_session(X, y, fs)

print("\nVerification:")
print(f"fdcc_select_band call count: {fdcc.fdcc_select_band.call_count}")
# Should be called 2 times (once for FTA, once for RG) before folds
if fdcc.fdcc_select_band.call_count == 2:
    print("SUCCESS: FDCC called exactly twice (once per feature type) for the session.")
else:
    print(f"FAILURE: FDCC called {fdcc.fdcc_select_band.call_count} times. Expected 2.")

# Reset config
config.FIXED_BEST_BAND_MODE = False
