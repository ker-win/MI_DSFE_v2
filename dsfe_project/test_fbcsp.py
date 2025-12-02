
import sys
import os
import numpy as np
import mne

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dsfe import config, train_eval, fbcsp

# Mock Data
n_trials = 20
n_channels = 22
n_times = 500
fs = 250.0

X = np.random.randn(n_trials, n_channels, n_times)
y = np.array([0, 1] * 10)

# Configure for test
config.FIXED_BEST_BAND_MODE = False
config.USE_FDCC = False # Disable FDCC to focus on FBCSP
config.USE_FTA = False
config.USE_RG = False
config.USE_FBCSP = True
config.FBCSP_BANDS = [(8, 12), (12, 16)] # 2 bands
config.N_CSP_COMPONENTS = 2 # 2 components per band
config.N_FOLDS_EVAL = 2
config.VERBOSE = True
config.USE_RELIEFF = False # Disable ReliefF to check raw dimension first

print("Testing FBCSP standalone...")
fbcsp_model = fbcsp.FBCSP(bands=config.FBCSP_BANDS, n_components=config.N_CSP_COMPONENTS)
fbcsp_model.fit(X, y, fs)
X_trans = fbcsp_model.transform(X, fs)
print(f"Transformed Shape: {X_trans.shape}")

expected_dim = len(config.FBCSP_BANDS) * config.N_CSP_COMPONENTS
if X_trans.shape == (n_trials, expected_dim):
    print(f"SUCCESS: Shape matches expected ({n_trials}, {expected_dim})")
else:
    print(f"FAILURE: Shape mismatch. Expected ({n_trials}, {expected_dim})")

print("\nTesting Integration in evaluate_session...")
# Enable fusion with ReliefF
config.USE_RELIEFF = True
config.RELIEFF_KEEP_RATIO = 0.5
try:
    acc = train_eval.evaluate_session(X, y, fs)
    print(f"Evaluation finished. Acc: {acc}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
