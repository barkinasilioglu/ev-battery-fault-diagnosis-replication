import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score
)

# =========================
# 1) LOAD DATA
# =========================

CSV_PATH = "EV_Battery_Fault_Diagnosis.csv"
df = pd.read_csv(CSV_PATH)

# =========================
# 2) LABEL MAPPING
# =========================

df["Fault Label"] = df["Fault Label"].map({
    "Normal": 0,
    "Warning": 1
})

# =========================
# 3) FEATURE SELECTION
# =========================

feature_cols = [
    "Voltage (V)",
    "Current (A)",
    "Temperature (°C)",
    "Motor Speed (RPM)",
    "Hall Code",
    "Estimated SOC (%)",
    "Ground Truth SOC (%)",
   
]

X = df[feature_cols].values
y = df["Fault Label"].values
y = y.astype(int)

print("Class distribution:", np.bincount(y))

# =========================
# 4) TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# 5) MODELS
# =========================

models = {
    "GaussianNB": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

# =========================
# 6) TRAIN + EVALUATE
# =========================

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "="*60)
    print("Model:", name)
    print("F1 Score (macro):", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_{name}.png", dpi=200)
    plt.close()

print("\nConfusion matrices saved.")

# =========================
# 7) PCA VISUALIZATION
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, label="Normal")
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.5, label="Fault")
plt.title("PCA (2D) Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=200)
plt.close()

print("PCA plot saved as pca_2d.png")
