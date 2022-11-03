from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

pipeline = Pipeline([
  ("scaler", StandardScaler()),
  ("svm_clf", SVC(probability=True, random_state=cfg.SEED))
])

gs_params = {
  "svm_clf__kernel": [
    "linear",
    "rbf"
  ],
  "svm_clf__C": [
    1e-05,
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
  ],
  "svm_clf__gamma": [
    "scale",
    "auto"
  ],
  "svm_clf__class_weight": [
    None,
    "balanced"
  ]
}