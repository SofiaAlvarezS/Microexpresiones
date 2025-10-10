# src/train_svm.py
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model(cfg):
    features_path = Path(cfg["features"]["output_file"])
    model_path = Path(cfg["model"]["output_path"])
    training_cfg = cfg["model"]["training"]

    test_size = training_cfg.get("test_size", 0.2)
    random_state = training_cfg.get("random_state", 42)
    class_weight = training_cfg.get("class_weight", "balanced")
    use_cv = training_cfg.get("use_cross_validation", False)
    cv_folds = training_cfg.get("cv_folds", 5)

    print(f"\n📂 Cargando características desde: {features_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de características: {features_path}")

    df = pd.read_csv(features_path)
    if "label" not in df.columns:
        raise ValueError("El archivo de características no contiene la columna 'label'.")

    print(f"✅ {len(df)} muestras cargadas con {df.shape[1]-1} características cada una.")

    X = df.drop(columns=["label"])
    y = df["label"]

    print("\n🧠 Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\n⚙️ Entrenando modelo SVM...")
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=random_state)
    )

    if use_cv:
        print(f"🔁 Validación cruzada ({cv_folds} folds)...")
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
        print(f"📊 Accuracy promedio CV: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    clf.fit(X_train, y_train)
    print("\n✅ Entrenamiento completado.")

    print("\n📈 Evaluando modelo en el conjunto de prueba...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("🧩 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")

    # Guardar métricas simples en un CSV opcional
    metrics_path = model_path.parent / "training_metrics.csv"
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(metrics_path, index=True)
    print(f"📄 Reporte guardado en: {metrics_path}")

if __name__ == "__main__":
    cfg = load_config()
    train_model(cfg)
