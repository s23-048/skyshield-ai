# train.py
import os

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from preprocess import load_bird_strike_data


def main():
    # 1. Load data
    print("Loading data...")
    df = load_bird_strike_data()

    feature_cols = ["AIRCRAFT", "PHASE_OF_FLIGHT", "SPECIES", "TIME_OF_DAY", "HEIGHT"]
    target_col = "DAMAGE_SEVERITY"

    X = df[feature_cols]
    y = df[target_col]

    categorical_features = ["AIRCRAFT", "PHASE_OF_FLIGHT", "SPECIES", "TIME_OF_DAY"]
    numeric_features = ["HEIGHT"]

    # 2. Define preprocessing + model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model),
        ]
    )

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    pipe.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating model...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # 5. Save pipeline
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "bird_strike_pipeline.pkl")
    joblib.dump(pipe, model_path)

    print(f"\nModel pipeline saved to: {model_path}")


if __name__ == "__main__":
    main()
