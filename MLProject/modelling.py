import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("Titanic Classification")

    mlflow.sklearn.autolog()

    df = pd.read_csv(
        "preprocessing/titanic_preprocessed.csv"
    )

    print("=" * 50)
    print("Dataset Preview")
    print("=" * 50)
    print(df.head())

    X = df.drop(
        columns=["Survived"]
    )

    y = df["Survived"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run():

        model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )


        joblib.dump(
            model,
            "model.pkl"
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        print("=" * 50)
        print(f"Accuracy : {accuracy:.4f}")
        print("=" * 50)

        print("Model berhasil disimpan sebagai best_model.pkl")
        print("Training selesai.")


if __name__ == "__main__":
    main()