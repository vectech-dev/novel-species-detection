from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from unknown.config.config import UnknownConfigLoader
cfg = UnknownConfigLoader()

pipeline = Pipeline([
  ("rfor_clf", RandomForestClassifier(random_state=cfg.SEED))
])

gs_params = {
  "rfor_clf__n_estimators": [
    30,
    50,
    100,
    200,
    500,
    700,
    1000
  ],
  "rfor_clf__max_features": [
    "auto",
    "sqrt",
    "log2"
  ],
  "rfor_clf__max_depth": [
    8,
    10,
    12,
    15,
    19
  ],
  "rfor_clf__criterion": [
    "gini",
    "entropy"
  ]
}
