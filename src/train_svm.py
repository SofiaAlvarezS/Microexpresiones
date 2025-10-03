# src/train_svm.py
import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # cargar configuración
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    features_file = cfg["features"]["output_file"]
    model_path = cfg["model"]["output_path"]

    # cargar features
    df = pd.read_csv(features_file)
    print(f"📊 Features cargadas: {len(df)} muestras, {df.shape[1]} columnas")

    # separar X e y (quitamos columnas no numéricas)
    drop_cols = [col for col in ["file", "label", "dataset"] if col in df.columns]
    X = df.drop(columns=drop_cols).values
    y = df["label"].values

    # config de entrenamiento
    params = cfg["model"].get("training", {})
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    class_weight = params.get("class_weight", "balanced")
    use_cv = params.get("use_cross_validation", False)
    cv_folds = params.get("cv_folds", 5)

    clf = SVC(kernel="linear", class_weight=class_weight, probability=True)

    if use_cv:
        print(f"🔄 Entrenando con cross-validation ({cv_folds} folds)...")
        scores = cross_val_score(clf, X, y, cv=cv_folds)
        print("CV scores:", scores)
        print("Mean accuracy:", scores.mean())
        clf.fit(X, y)
    else:
        print(f"🧪 Hold-out split {int((1-test_size)*100)}/{int(test_size*100)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("📋 Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

    # guardar modelo
    joblib.dump(clf, model_path)
    print(f"✅ Modelo guardado en {model_path}")

if __name__ == "__main__":
    main()
