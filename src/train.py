import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def build_pipeline(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return clf

def main(data_path: str, out_path: str):
    df = pd.read_csv(data_path)
    y = df["HeartDisease"]
    X = df.drop(columns=["HeartDisease"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = build_pipeline(X_train)

    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_alpha": [0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 10.0, 100.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)

    print("Saved model to:", out_path)
    print("Best params:", search.best_params_)
    print("Best CV ROC AUC:", search.best_score_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path do pliku heart.csv")
    parser.add_argument("--out", required=True, help="Gdzie zapisać model .joblib")
    args = parser.parse_args()
    main(args.data, args.out)