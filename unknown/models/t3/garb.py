import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

class BinaryGaussianArbitrationClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, unsure_lower_bound=0.1, unsure_upper_bound=0.9,
        n_components=7, n_init=10, unknown_frac=0.04,
        thresholding_method='percentile', random_state=None):
        self.unsure_lower_bound = unsure_lower_bound
        self.unsure_upper_bound = unsure_upper_bound
        self.unknown_frac = unknown_frac
        self.random_state = random_state
        self.n_components = n_components
        self.n_init = n_init
        assert thresholding_method in ['percentile', 'std'], "Invalid thresholding method"
        self.thresholding_method = thresholding_method

    def fit(self, X, y):
        x_train = X
        y_train = y
        df = pd.DataFrame(x_train)
        df['y'] = y_train
        df_known = df[df.y == 0]
        df_unknown = df[df.y == 1].sample(frac=self.unknown_frac, random_state=self.random_state)

        df_all = pd.concat([df_known, df_unknown], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        y_train_mk = df_all.pop('y')
        x_train_mk = df_all.to_numpy()

        # define model
        gm = GaussianMixture(n_components=self.n_components, n_init=self.n_init, random_state=self.random_state)
        gm.fit(x_train_mk)
        densities = gm.score_samples(x_train_mk)
        
        if self.thresholding_method == 'std':
            self.density_threshold = np.mean(densities) - (1 * np.std(densities))
        elif self.thresholding_method == 'percentile':
            self.density_threshold = np.percentile(densities, int(self.unknown_frac * 100))
        else:
            raise NotImplementedError("Invalid thresholding method")
        self.gm = gm

    def predict(self, X):
        if self.unsure_lower_bound == 0 and self.unsure_upper_bound == 1:
            scores = self.gm.score_samples(X)
            return (scores < self.density_threshold)*1
        else:
            df = pd.DataFrame(X)
            df['probas'] = np.apply_along_axis(lambda row: sum(row)/len(row), axis=1, arr=X)
            df['preds'] = np.apply_along_axis(lambda row: int((sum(row)/len(row)) > 0.5), axis=1, arr=X)

            df_unsure = df[(df.probas <= self.unsure_upper_bound) & (df.probas >= self.unsure_lower_bound)]
            x_unsure = df_unsure.drop(columns=['probas', 'preds']).to_numpy()

            scores = self.gm.score_samples(x_unsure)
            preds_unsure = (scores < self.density_threshold)*1

            df.iloc[df_unsure.index, df.columns.get_loc("preds")] = preds_unsure

            return df.preds.to_numpy()
    
    def save(self, fname: str):
        serialized = {
            'gm': self.gm,
            'threshold': self.density_threshold,
            "unsure_lower_bound": self.unsure_lower_bound,
            "unsure_upper_bound": self.unsure_upper_bound,
            "n_components": self.n_components,
            "n_init": self.n_init,
            "unknown_frac": self.unknown_frac,
            "thresholding_method": self.thresholding_method,
            "random_state": self.random_state
        }
        joblib.dump(serialized, fname)
    
    @classmethod
    def load(cls, fname: str):
        serialized = joblib.load(fname)
        new_class = cls(
                        unsure_lower_bound=serialized["unsure_lower_bound"],
                        unsure_upper_bound=serialized["unsure_upper_bound"],
                        n_components=serialized["n_components"],
                        n_init=serialized["n_init"],
                        unknown_frac=serialized["unknown_frac"],
                        thresholding_method=serialized["thresholding_method"],
                        random_state=serialized["random_state"])
        new_class.gm = serialized['gm']
        new_class.density_threshold = serialized.threshold
        return new_class

class GarbPipeline(Pipeline):
    def save(self, fname: str):
        self.named_steps['garb_clf'].save(fname)


pipeline = GarbPipeline([
  ("garb_clf", BinaryGaussianArbitrationClassifier(random_state=cfg.SEED))
])

gs_params = {
    "garb_clf__unsure_lower_bound": [0.1, 0.2, 0.3, 0.4, 0.5],
    "garb_clf__unsure_upper_bound": [0.9, 0.8, 0.7, 0.6],
    "garb_clf__unknown_frac": [0.04, 0.1]
}