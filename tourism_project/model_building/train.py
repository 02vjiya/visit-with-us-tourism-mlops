
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv")["ProdTaken"]
y_test = pd.read_csv("ytest.csv")["ProdTaken"]

numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical = X_train.select_dtypes(exclude=["number"]).columns.tolist()
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric),
    ("categorical", categorical_pipe, categorical)
])
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1))
])
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 12],
    "model__min_samples_leaf": [1, 3]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=1)
search.fit(X_train, y_train)
best_model = search.best_estimator_
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob))
}
print("Best parameters:", search.best_params_)
print("Best CV F1:", round(search.best_score_, 4))
print("Test metrics:", json.dumps(metrics, indent=2))
print(classification_report(y_test, pred, zero_division=0))

# Optional MLflow tracking; model training remains reproducible without an external server.
try:
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("visit-with-us-tourism")
    with mlflow.start_run():
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
    print("MLflow run logged to ./mlruns")
except ImportError:
    print("MLflow is not installed in this environment; install requirements to enable tracking.")
except Exception as exc:
    print(f"MLflow tracking was skipped: {exc}")

Path("tourism_project/deployment").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, "tourism_project/deployment/tourism_model.joblib")
with open("tourism_project/deployment/model_metrics.json", "w") as f:
    json.dump({"best_params": search.best_params_, "metrics": metrics}, f, indent=2)
print("Saved model: tourism_project/deployment/tourism_model.joblib")
