
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from . import config, preprocess, features, fdcc, fusion, ensemble, fbcsp

def run_single_fold(X_train, y_train, X_test, y_test, fs, fixed_band_fta=None, fixed_band_rg=None):
    """
    Run DSFE pipeline for a single fold.
    """
    # 1. Band Selection (FDCC)
    # We need to determine bands for FTA and RG
    if config.USE_FDCC:
        if config.VERBOSE: print("  Running FDCC...")
        
        # FTA Band
        if fixed_band_fta is not None:
            band_fta = fixed_band_fta
            if config.VERBOSE: print(f"    Using Fixed FTA Band: {band_fta}")
        elif config.USE_FTA:
            band_fta = fdcc.fdcc_select_band(X_train, y_train, fs, 'fta')
        else:
            band_fta = None

        # RG Band
        if fixed_band_rg is not None:
            band_rg = fixed_band_rg
            if config.VERBOSE: print(f"    Using Fixed RG Band: {band_rg}")
        elif config.USE_RG:
            band_rg = fdcc.fdcc_select_band(X_train, y_train, fs, 'rg')
        else:
            band_rg = None
            
        if config.VERBOSE: print(f"  Best Bands - FTA: {band_fta}, RG: {band_rg}")
    else:
        band_fta = config.FDCC_FREQ_RANGE
        band_rg = config.FDCC_FREQ_RANGE
        
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
        
    # FBCSP
    F_fbcsp_train = None
    F_fbcsp_test = None
    
    if config.USE_FBCSP:
        if config.VERBOSE: print("    Running FBCSP...")
        fbcsp_model = fbcsp.FBCSP(bands=config.FBCSP_BANDS, n_components=config.N_CSP_COMPONENTS)
        # Fit on TRAIN only
        fbcsp_model.fit(X_train, y_train, fs)
        
        # Transform both
        F_fbcsp_train = fbcsp_model.transform(X_train, fs)
        F_fbcsp_test = fbcsp_model.transform(X_test, fs)
        
        feature_sets_train.append(F_fbcsp_train)
        feature_sets_test.append(F_fbcsp_test)

    # Fusion (FTA + RG + FBCSP + ReliefF)
    # Collect all available features
    features_to_fuse_train = []
    features_to_fuse_test = []
    
    if F_fta_train is not None:
        features_to_fuse_train.append(F_fta_train)
        features_to_fuse_test.append(F_fta_test)
        
    if F_rg_train is not None:
        features_to_fuse_train.append(F_rg_train)
        features_to_fuse_test.append(F_rg_test)
        
    if F_fbcsp_train is not None:
        features_to_fuse_train.append(F_fbcsp_train)
        features_to_fuse_test.append(F_fbcsp_test)
        
    if features_to_fuse_train:
        # Fuse
        F_fused_train, idx_sel = fusion.fuse_features(features_to_fuse_train, y_train)
        
        # Apply selection to test
        # Concatenate test features in same order
        F_all_test = np.concatenate(features_to_fuse_test, axis=1)
        F_fused_test = F_all_test[:, idx_sel]
        
        feature_sets_train.append(F_fused_train)
        feature_sets_test.append(F_fused_test)
        
    # 3. Classification
    if not feature_sets_train:
        raise ValueError("No features enabled (FTA, RG, FBCSP are all False)")
        
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
    
    # --- Fixed Best Band Mode ---
    fixed_band_fta = None
    fixed_band_rg = None
    
    if config.FIXED_BEST_BAND_MODE and config.USE_FDCC:
        print("  [Fixed Best Band Mode] Calculating best bands on entire session data...")
        if config.USE_FTA:
            fixed_band_fta = fdcc.fdcc_select_band(X, y, fs, 'fta')
            print(f"    -> Fixed FTA Band: {fixed_band_fta}")
        if config.USE_RG:
            fixed_band_rg = fdcc.fdcc_select_band(X, y, fs, 'rg')
            print(f"    -> Fixed RG Band: {fixed_band_rg}")
            
    # ----------------------------
    
    cv = StratifiedKFold(n_splits=config.N_FOLDS_EVAL, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if config.VERBOSE: print(f"Fold {i+1}/{config.N_FOLDS_EVAL}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        score = run_single_fold(X_train, y_train, X_test, y_test, fs, 
                                fixed_band_fta=fixed_band_fta, 
                                fixed_band_rg=fixed_band_rg)
        scores.append(score)
        if config.VERBOSE: print(f"  Acc: {score:.4f}")
        
    return np.mean(scores)
