import os, json, urllib.request, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR = os.path.join(BASE_DIR, "artifacts")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
DATA_PATH = os.path.join(DATA_DIR, "titanic.csv")

def load_data():
    if not os.path.exists(DATA_PATH):
        print("Mengunduh dataset Titanic...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        except Exception as exc:
            raise RuntimeError(
                "Dataset belum tersedia dan unduhan gagal. Periksa internet, lalu jalankan ulang. "
                "Alternatif: simpan file titanic.csv ke folder data."
            ) from exc
    df = pd.read_csv(DATA_PATH)
    required = ["survived", "pclass", "sex", "age", "fare"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing}")
    return df[required].copy()

def main():
    df = load_data()
    print("\\n=== 5 DATA PERTAMA ===")
    print(df.head())
    print("\\n=== INFORMASI DATA ===")
    print(df.info())
    print("\\n=== NILAI KOSONG ===")
    print(df.isnull().sum())

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="survived")
    plt.title("Distribusi Kelangsungan Hidup Penumpang Titanic")
    plt.xlabel("Survived (0 = Tidak, 1 = Ya)")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "distribusi_target.png"), dpi=150)
    plt.close()

    # EDA sederhana sesuai fitur pada modul
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=df, x="sex", hue="survived", ax=axes[0])
    axes[0].set_title("Jenis Kelamin dan Kelangsungan Hidup")
    sns.boxplot(data=df, x="survived", y="age", ax=axes[1])
    axes[1].set_title("Usia berdasarkan Kelangsungan Hidup")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "eda_fitur.png"), dpi=150)
    plt.close()

    X = df[["pclass", "sex", "age", "fare"]]
    y = df["survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    numeric_features = ["pclass", "age", "fare"]
    categorical_features = ["sex"]
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    model_a = Pipeline([
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeClassifier(random_state=42))
    ])
    model_b = Pipeline([
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=42))
    ])
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    results = {}
    for name, model in [("Model A - Tanpa Pruning", model_a),
                        ("Model B - Pruning max_depth=3", model_b)]:
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        results[name] = {
            "akurasi_latih": round(float(accuracy_score(y_train, pred_train)), 4),
            "akurasi_uji": round(float(accuracy_score(y_test, pred_test)), 4),
            "classification_report": classification_report(
                y_test, pred_test, target_names=["Tidak Selamat", "Selamat"], output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, pred_test).tolist()
        }
        print(f"\\n=== {name} ===")
        print("Akurasi data latih:", results[name]["akurasi_latih"])
        print("Akurasi data uji:", results[name]["akurasi_uji"])
        print(classification_report(y_test, pred_test, target_names=["Tidak Selamat", "Selamat"]))
        print("Confusion Matrix:\\n", confusion_matrix(y_test, pred_test))

    # Simpan model B untuk aplikasi Flask
    joblib.dump(model_b, os.path.join(ART_DIR, "decision_tree_titanic.joblib"))
    with open(os.path.join(ART_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    tree = model_b.named_steps["tree"]
    feature_names = model_b.named_steps["preprocessor"].get_feature_names_out()
    plt.figure(figsize=(18, 9))
    plot_tree(tree, feature_names=feature_names,
              class_names=["Tidak Selamat", "Selamat"],
              filled=True, rounded=True, fontsize=9)
    plt.title("Decision Tree Titanic (Pruning max_depth=3)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "pohon_keputusan.png"), dpi=160)
    plt.close()

    cm = confusion_matrix(y_test, model_b.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Selamat", "Selamat"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Model B")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    importance = pd.DataFrame({
        "Fitur": feature_names,
        "Pentingnya": tree.feature_importances_
    }).sort_values("Pentingnya", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, x="Pentingnya", y="Fitur")
    plt.title("Feature Importance - Model B")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    with open(os.path.join(ART_DIR, "aturan_pohon.txt"), "w", encoding="utf-8") as f:
        f.write(export_text(tree, feature_names=list(feature_names)))
    print("\\nSelesai. Model, metrik, dan grafik tersimpan.")
    print("Buka aplikasi dengan: python app.py")

if __name__ == "__main__":
    main()
