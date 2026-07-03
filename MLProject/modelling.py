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

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Titanic Classification")

    mlflow.sklearn.autolog()

    df = pd.read_csv(
        "titanic_preprocessed.csv"
    )


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

    

    model = RandomForestClassifier(
            random_state=42,
            n_estimators=50,
            n_jobs=-1
    )

    model.fit(
            X_train,
            y_train
    )


    joblib.dump(model, "best_model.pkl")

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
    print("Training selesai.")

if __name__ == "__main__":
    main()