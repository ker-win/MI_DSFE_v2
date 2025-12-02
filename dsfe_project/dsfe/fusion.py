
import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import config

def reliefF_impl(X, y, n_neighbors=20, n_features_to_keep=10):
    """
    Custom implementation of ReliefF algorithm.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    classes = np.unique(y)
    
    # Normalize X (Min-Max)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    range_X = X_max - X_min
    # Avoid division by zero
    range_X[range_X == 0] = 1.0
    X_norm = (X - X_min) / range_X
    
    # Group data by class
    X_by_class = {c: X_norm[y == c] for c in classes}
    p_classes = {c: len(X_by_class[c]) / n_samples for c in classes}
    
    # Precompute NearestNeighbors for each class
    nbrs_by_class = {}
    for c in classes:
        n_c = len(X_by_class[c])
        if n_c > n_neighbors:
            # Use Manhattan distance (L1) as per ReliefF usually
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan').fit(X_by_class[c])
            nbrs_by_class[c] = nbrs
        else:
            nbrs_by_class[c] = None
            
    # Iterate over all samples
    for i in range(n_samples):
        R = X_norm[i]
        c_R = y[i]
        
        # 1. Find k nearest hits (same class)
        hits = []
        if nbrs_by_class[c_R]:
            # Find k+1 because the first one is itself
            dists, ind = nbrs_by_class[c_R].kneighbors([R], n_neighbors=n_neighbors+1)
            hits_idx = ind[0][1:]
            hits = X_by_class[c_R][hits_idx]
        else:
            # Fallback: use all others in class
            others = X_by_class[c_R]
            # Ideally remove self, but for simplicity use all if small
            hits = others
        
        if len(hits) > 0:
            diff_hits = np.mean(np.abs(R - hits), axis=0)
        else:
            diff_hits = np.zeros(n_features)
            
        # 2. Find k nearest misses (other classes)
        diff_misses_weighted = np.zeros(n_features)
        sum_prob_miss = 0.0
        
        for c in classes:
            if c == c_R: continue
            
            misses = []
            if nbrs_by_class[c]:
                dists, ind = nbrs_by_class[c].kneighbors([R], n_neighbors=n_neighbors)
                misses = X_by_class[c][ind[0]]
            else:
                misses = X_by_class[c]
            
            if len(misses) > 0:
                diff_miss = np.mean(np.abs(R - misses), axis=0)
                prob = p_classes[c]
                diff_misses_weighted += prob * diff_miss
                sum_prob_miss += prob
                
        if sum_prob_miss > 0:
            diff_misses_weighted /= sum_prob_miss
            
        # Update weights
        weights += -diff_hits + diff_misses_weighted
        
    # Select top features
    idx_sorted = np.argsort(weights)[::-1]
    idx_selected = idx_sorted[:n_features_to_keep]
    
    return idx_selected, weights

def fuse_features(F_fta, F_rg, y):
    """
    Concatenate features and apply ReliefF if enabled.
    """
    # Concatenate
    F_all = np.concatenate([F_fta, F_rg], axis=1)
    n_features = F_all.shape[1]
    
    if not config.USE_RELIEFF:
        return F_all, np.arange(n_features)
        
    n_keep = int(n_features * config.RELIEFF_KEEP_RATIO)
    if n_keep < 1: n_keep = 1
    
    print(f"Running ReliefF... (Input dim: {n_features}, Keep: {n_keep})")
    idx_sel, _ = reliefF_impl(F_all, y, 
                              n_neighbors=config.N_NEIGHBORS_RELIEFF, 
                              n_features_to_keep=n_keep)
                              
    F_refined = F_all[:, idx_sel]
    return F_refined, idx_sel
