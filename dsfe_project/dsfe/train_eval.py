
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from . import config, preprocess, features, fdcc, fusion, ensemble

def run_single_fold(X_train, y_train, X_test, y_test, fs):
    """
    Run DSFE pipeline for a single fold.
    """
    # 1. Band Selection (FDCC)
    # We need to determine bands for FTA and RG
    if config.USE_FDCC:
        if config.VERBOSE: print("  Running FDCC...")
        band_fta = fdcc.fdcc_select_band(X_train, y_train, fs, 'fta') if config.USE_FTA else None
        band_rg = fdcc.fdcc_select_band(X_train, y_train, fs, 'rg') if config.USE_RG else None
        if config.VERBOSE: print(f"  Best Bands - FTA: {band_fta}, RG: {band_rg}")
    else:
        band_fta = config.FREQ_RANGE
        band_rg = config.FREQ_RANGE
        
    # 2. Feature Extraction
    feature_sets_train = []
    feature_sets_test = []
    
    # Keep track of which sets we have for the ensemble
    # Paper uses: [FTA, RG, Fused]
    
    F_fta_train = None
    F_fta_test = None
    F_rg_train = None
    F_rg_test = None
    
    # FTA
    if config.USE_FTA:
        F_fta_train = features.compute_fta_features(X_train, fs, band_fta)
        F_fta_test = features.compute_fta_features(X_test, fs, band_fta)
        feature_sets_train.append(F_fta_train)
        feature_sets_test.append(F_fta_test)
        
    # RG
    if config.USE_RG:
        # Filter first
        X_train_rg = fdcc.bandpass_data(X_train, fs, band_rg[0], band_rg[1])
        X_test_rg = fdcc.bandpass_data(X_test, fs, band_rg[0], band_rg[1])
        
        # Compute P_G on train
        F_rg_train, P_G = features.compute_rg_features(X_train_rg, fs)
        # Apply P_G on test
        F_rg_test = features.compute_rg_features_with_mean(X_test_rg, fs, P_G)
        
        feature_sets_train.append(F_rg_train)
        feature_sets_test.append(F_rg_test)
        
    # Fusion (FTA + RG + ReliefF)
    # Only if we have something to fuse
    if F_fta_train is not None and F_rg_train is not None:
        # We have both
        F_fused_train, idx_sel = fusion.fuse_features(F_fta_train, F_rg_train, y_train)
        
        # Apply selection to test
        F_all_test = np.concatenate([F_fta_test, F_rg_test], axis=1)
        F_fused_test = F_all_test[:, idx_sel]
        
        feature_sets_train.append(F_fused_train)
        feature_sets_test.append(F_fused_test)
    elif F_fta_train is not None:
        # Only FTA, "Fusion" is just FTA (maybe with ReliefF?)
        # Paper implies Fusion is the 3rd branch.
        # If we miss one, we might skip the 3rd branch or just duplicate.
        # Let's skip adding a specific "Fused" branch if we don't have multiple types,
        # UNLESS ReliefF is requested on the single type.
        # But for simplicity, if we only have one type, we just use that type.
        pass
    elif F_rg_train is not None:
        pass
        
    # 3. Classification
    if not feature_sets_train:
        raise ValueError("No features enabled (USE_FTA and USE_RG are both False)")
        
    if config.USE_ENSEMBLE:
        ens = ensemble.DSFEEnsemble()
        for F_tr in feature_sets_train:
            model = ensemble.FeatureSetModel()
            model.fit(F_tr, y_train)
            ens.add_model(model)
            
        y_pred = ens.predict(feature_sets_test)
    else:
        # Single Classifier (SVM)
        # Use the last feature set (likely the most processed/combined one)
        clf = SVC(kernel='rbf', gamma='scale', random_state=config.RANDOM_STATE)
        clf.fit(feature_sets_train[-1], y_train)
        y_pred = clf.predict(feature_sets_test[-1])
        
    return accuracy_score(y_test, y_pred)

def evaluate_session(X, y, fs):
    """
    10-fold CV on a session.
    """
    cv = StratifiedKFold(n_splits=config.N_FOLDS_EVAL, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if config.VERBOSE: print(f"Fold {i+1}/{config.N_FOLDS_EVAL}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        score = run_single_fold(X_train, y_train, X_test, y_test, fs)
        scores.append(score)
        if config.VERBOSE: print(f"  Acc: {score:.4f}")
        
    return np.mean(scores)
