
# dsfe_project/dsfe/config.py

# ===== System Parameters =====
RANDOM_STATE = 42
VERBOSE = True

# ===== Data Parameters =====
FS = 250.0  # Sampling rate (User specified 250Hz)

# ===== Preprocessing =====
# Trial Structure:
#   0.0s - 2.0s: Cue (Visual Cue)
#   2.0s - 6.0s: MI (Motor Imagery)
#   6.0s - 10.0s: Rest
#
# Select the time window for analysis (in seconds relative to trial start):
# Example (Paper): 0.0 - 0.85 (Cue onset)
# Example (MI):    2.0 - 6.0  (Full MI)
# Example (MI start): 2.0 - 3.0
EPOCH_TMIN = 3.0   # Start time
EPOCH_TMAX = 4.0  # End time
LOWPASS_FREQ = 30.0 # Paper: 30Hz lowpass

# ===== FDCC (Feature-Dependent Correlation Coefficient) =====
FDCC_FREQ_RANGE = (0.0, 30.0) # Search range
SUBBAND_WIDTH = 2.0           # 2Hz step
T_CANDIDATES = [3, 4, 5, 6, 7, 8] # Number of top subbands to consider
N_FOLDS_FDCC = 5

# ===== ReliefF =====
RELIEFF_KEEP_RATIO = 0.25
N_NEIGHBORS_RELIEFF = 20

# ===== Ensemble Weights =====
# Paper: SVM=0.16, RF=0.09, NB=0.08
SVM_WEIGHT = 0.16
RF_WEIGHT  = 0.09
NB_WEIGHT  = 0.08
N_TREES_RF = 100

# ===== Evaluation =====
N_FOLDS_EVAL = 5
MERGE_RUNS = False # If True, merge all runs for a subject/modality before evaluation

# ===== Ablation Study Flags =====
# Set these to False to disable specific components
USE_FTA = True       # Fourier Transform Amplitudes
USE_RG = False        # Riemannian Geometry features
USE_FDCC = True     # Feature-Dependent Band Selection (if False, use full 0-30Hz)
USE_RELIEFF = True   # Feature Selection (if False, use all features)
USE_ENSEMBLE = True  # Ensemble Learning (if False, use single SVM)
FIXED_BEST_BAND_MODE = False    # If True, select best band using FDCC on whole dataset before CV
USE_FBCSP = True     # Filter Bank Common Spatial Patterns
FBCSP_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
N_CSP_COMPONENTS = 2 # Number of components per band (reduced to 2 for 3-channel data)
CSP_REG = 0.001      # Regularization for CSP (to avoid LinAlgError)
CSP_RANK = None      # Rank for CSP (None = estimate)
