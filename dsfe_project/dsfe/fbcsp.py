
import numpy as np
import mne
from mne.decoding import CSP
from . import config

class FBCSP:
    def __init__(self, bands=None, n_components=4):
        self.bands = bands if bands else config.FBCSP_BANDS
        self.n_components = n_components
        self.csps = {} # Dictionary to store CSP objects for each band

    def fit(self, X, y, fs):
        """
        Fit CSP filters for each frequency band.
        X: (n_trials, n_channels, n_times)
        y: (n_trials,)
        """
        for band in self.bands:
            f_low, f_high = band
            # Filter data
            # Note: We use a copy to avoid modifying original X in place if filter works in place
            # MNE filter_data returns a copy by default usually, but let's be safe.
            # However, filtering inside the loop for every band is expensive.
            # But FBCSP requires it.
            
            # We use mne.filter.filter_data
            # verbose=False to reduce clutter
            X_filt = mne.filter.filter_data(X, fs, l_freq=f_low, h_freq=f_high, verbose=False)
            
            # Ensure float64 and contiguous for MNE CSP
            # MNE requires float64 and will error if copy=None and casting is needed.
            # We explicitly cast and copy to be safe.
            X_filt = np.ascontiguousarray(X_filt, dtype=np.float64)
                
            csp = CSP(n_components=self.n_components, reg=config.CSP_REG, rank=config.CSP_RANK, log=True, norm_trace=False)
            csp.fit(X_filt, y)
            self.csps[band] = csp
            
        return self

    def transform(self, X, fs):
        """
        Apply CSP filters and extract log-variance features.
        Returns: (n_trials, n_features) where n_features = n_bands * n_components
        """
        features_list = []
        
        for band in self.bands:
            if band not in self.csps:
                raise ValueError(f"Band {band} not fitted.")
            
            f_low, f_high = band
            X_filt = mne.filter.filter_data(X, fs, l_freq=f_low, h_freq=f_high, verbose=False)
            
            # Ensure float64 and contiguous
            X_filt = np.ascontiguousarray(X_filt, dtype=np.float64)
            
            csp = self.csps[band]
            # transform returns (n_trials, n_components)
            feats = csp.transform(X_filt)
            features_list.append(feats)
            
        # Concatenate all bands: (n_trials, n_bands * n_components)
        return np.concatenate(features_list, axis=1)
