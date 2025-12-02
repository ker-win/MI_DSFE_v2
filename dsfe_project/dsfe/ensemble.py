
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from . import config

class FeatureSetModel:
    """
    Trains 3 classifiers (SVM, RF, NB) on a single feature set.
    """
    def __init__(self):
        self.svm = SVC(kernel='rbf', gamma='scale', random_state=config.RANDOM_STATE)
        self.rf = RandomForestClassifier(n_estimators=config.N_TREES_RF, random_state=config.RANDOM_STATE)
        self.nb = GaussianNB()
        self.fitted = False
        
    def fit(self, X, y):
        self.svm.fit(X, y)
        self.rf.fit(X, y)
        self.nb.fit(X, y)
        self.fitted = True
        
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        return {
            'svm': self.svm.predict(X),
            'rf': self.rf.predict(X),
            'nb': self.nb.predict(X)
        }

class DSFEEnsemble:
    """
    Weighted voting ensemble across multiple feature sets.
    """
    def __init__(self):
        self.models = [] # List of FeatureSetModel
        
    def add_model(self, model):
        self.models.append(model)
        
    def predict(self, X_list):
        """
        X_list: List of feature matrices, one for each added model.
        """
        if not self.models:
            raise RuntimeError("No models added to ensemble")
            
        n_samples = X_list[0].shape[0]
        
        # We need to aggregate votes for each sample
        final_preds = []
        
        # Pre-calculate predictions for all models
        all_model_preds = []
        for i, model in enumerate(self.models):
            all_model_preds.append(model.predict(X_list[i]))
            
        for i in range(n_samples):
            scores = {}
            
            for preds in all_model_preds:
                # SVM
                p = preds['svm'][i]
                scores[p] = scores.get(p, 0) + config.SVM_WEIGHT
                
                # RF
                p = preds['rf'][i]
                scores[p] = scores.get(p, 0) + config.RF_WEIGHT
                
                # NB
                p = preds['nb'][i]
                scores[p] = scores.get(p, 0) + config.NB_WEIGHT
                
            # Find class with max score
            best_class = max(scores, key=scores.get)
            final_preds.append(best_class)
            
        return np.array(final_preds)
