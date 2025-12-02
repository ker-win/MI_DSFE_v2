
import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from . import config, features, preprocess

def get_subbands(f_min, f_max, width):
    """Generate list of subbands (start, end)."""
    bands = []
    curr = f_min
    while curr + width <= f_max:
        bands.append((curr, curr + width))
        curr += width # Non-overlapping? Prompt says "0-2, 2-4..." so yes.
    return bands

def bandpass_data(X, fs, f_low, f_high):
    """Apply bandpass filter."""
    # Ensure f_low is not 0 for bandpass, or use lowpass/highpass
    if f_low <= 0:
        return mne.filter.filter_data(X, fs, l_freq=None, h_freq=f_high, verbose=False)
    else:
        return mne.filter.filter_data(X, fs, l_freq=f_low, h_freq=f_high, verbose=False)

def compute_correlation_scores(F, y):
    """
    Compute average absolute Pearson correlation between features and labels.
    F: (n_samples, n_features)
    y: (n_samples,)
    Returns: scalar score (mean |r|)
    """
    # Vectorized correlation
    n = F.shape[0]
    if n < 2:
        return 0.0
        
    # Center data
    F_centered = F - F.mean(axis=0)
    y_centered = y - y.mean()
    
    # Denominator
    F_std = F.std(axis=0)
    y_std = y.std()
    
    # Avoid division by zero
    valid_idx = (F_std > 1e-10)
    if y_std < 1e-10:
        return 0.0
        
    corrs = np.zeros(F.shape[1])
    
    # Covariance
    cov = (F_centered.T @ y_centered) / n
    
    corrs[valid_idx] = cov[valid_idx] / (F_std[valid_idx] * y_std)
    
    return np.mean(np.abs(corrs))

def merge_adjacent_bands(bands):
    """
    Merge adjacent bands.
    bands: list of tuples [(f1, f2), (f3, f4), ...]
    Returns: list of merged tuples
    """
    if not bands:
        return []
    
    # Sort by start freq
    sorted_bands = sorted(bands, key=lambda x: x[0])
    merged = []
    if not sorted_bands:
        return []
        
    curr_start, curr_end = sorted_bands[0]
    
    for i in range(1, len(sorted_bands)):
        next_start, next_end = sorted_bands[i]
        # If adjacent (or overlapping, though prompt implies adjacent like 0-2, 2-4)
        if next_start <= curr_end + 1e-5: # Tolerance
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged

def select_best_band(X, y, fs, feature_type, candidate_bands):
    """
    From a list of candidate bands, select the one with highest score.
    """
    best_score = -1
    best_band = None
    
    for band in candidate_bands:
        # Extract features
        if feature_type == 'fta':
            F = features.compute_fta_features(X, fs, band)
        elif feature_type == 'rg':
            # Filter X first
            X_band = bandpass_data(X, fs, band[0], band[1])
            F, _ = features.compute_rg_features(X_band, fs) # Ignore P_G here
        
        score = compute_correlation_scores(F, y)
        if score > best_score:
            best_score = score
            best_band = band
            
    return best_band

def fdcc_select_band(X, y, fs, feature_type):
    """
    Main FDCC algorithm.
    Returns: best_band (f1, f2)
    """
    f_start, f_end = config.FDCC_FREQ_RANGE
    h = config.SUBBAND_WIDTH
    
    # 1. Generate subbands
    subbands = get_subbands(f_start, f_end, h)
    
    # 2. Calculate score for each subband
    subband_scores = []
    for band in subbands:
        if feature_type == 'fta':
            F = features.compute_fta_features(X, fs, band)
        elif feature_type == 'rg':
            X_band = bandpass_data(X, fs, band[0], band[1])
            F, _ = features.compute_rg_features(X_band, fs)
        
        score = compute_correlation_scores(F, y)
        subband_scores.append((score, band))
    
    # Sort by score descending
    subband_scores.sort(key=lambda x: x[0], reverse=True)
    sorted_bands = [x[1] for x in subband_scores]
    
    # 3. Cross-validation to find best T
    best_cv_acc = -1
    best_band_final = None
    
    # If T is larger than available bands, cap it
    valid_Ts = [t for t in config.T_CANDIDATES if t <= len(subbands)]
    if not valid_Ts:
        valid_Ts = [len(subbands)]
        
    for T in valid_Ts:
        # Top T bands
        top_T_bands = sorted_bands[:T]
        
        # Merge
        candidate_bands = merge_adjacent_bands(top_T_bands)
        
        # Select best candidate (using whole training set X, y? 
        # Paper says: "Select the one with max avg |r|". 
        # This step is deterministic given X, y.
        best_band_T = select_best_band(X, y, fs, feature_type, candidate_bands)
        
        # Evaluate this band using CV
        # Extract features for this band
        if feature_type == 'fta':
            F = features.compute_fta_features(X, fs, best_band_T)
        elif feature_type == 'rg':
            X_band = bandpass_data(X, fs, best_band_T[0], best_band_T[1])
            F, _ = features.compute_rg_features(X_band, fs)
            
        # 5-fold CV
        cv = StratifiedKFold(n_splits=config.N_FOLDS_FDCC, shuffle=True, random_state=config.RANDOM_STATE)
        accs = []
        for train_idx, val_idx in cv.split(F, y):
            # For RG, we should technically re-compute P_G on train_idx only?
            # The prompt's simplified FDCC code doesn't explicitly say.
            # But for standard CV, yes.
            # However, F is already extracted.
            # If we extracted RG on whole X, we leaked information (P_G computed on whole X).
            # But doing RG inside CV loop is very expensive.
            # Given the prompt's simplified snippet:
            # "F = extract_feature_for_band(X, best_band_T)" -> "clf.fit..."
            # It implies extracting once.
            # I will follow the prompt's simplification to avoid extreme slowness.
            
            clf = SVC(kernel='rbf', gamma='scale')
            clf.fit(F[train_idx], y[train_idx])
            acc = clf.score(F[val_idx], y[val_idx])
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_band_final = best_band_T
            
    return best_band_final
