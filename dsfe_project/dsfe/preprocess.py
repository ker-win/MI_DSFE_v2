
import numpy as np
import mne
from . import config

def apply_average_reference(X):
    """
    Apply average reference to EEG data.
    X: (n_trials, n_channels, n_times)
    """
    # Subtract the mean across channels for each time point
    return X - np.mean(X, axis=1, keepdims=True)

def apply_lowpass(X, fs, f_high):
    """
    Apply lowpass filter.
    X: (n_trials, n_channels, n_times)
    fs: sampling rate
    f_high: cutoff frequency
    """
    # Use MNE's filter_data which handles numpy arrays
    # verbose=False to suppress output
    return mne.filter.filter_data(X, fs, l_freq=None, h_freq=f_high, verbose=False)

def preprocess_pipeline(X, fs):
    """
    Full preprocessing pipeline for DSFE.
    1. Average Reference
    2. Lowpass Filter (30Hz)
    """
    X_ref = apply_average_reference(X)
    X_filt = apply_lowpass(X_ref, fs, config.LOWPASS_FREQ)
    return X_filt
