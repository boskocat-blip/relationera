
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "raw" / "divorce.csv"

# Läser in data
df = pd.read_csv(DATA_FILE, sep=";")

if df.shape[1] == 1:
    df = pd.read_csv(DATA_FILE, sep=",")

df.columns = df.columns.str.strip()


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Dataset:", df.shape)
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# --------------------------------------------------
# 2. Random Forest
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Skapa och träna modellen
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)

random_forest.fit(X_train, y_train)

# Gör prediktioner på testdatan
y_pred_rf = random_forest.predict(X_test)

# Utvärdera modellen
print("\n=== Random Forest ===")
print("Träffsäkerhet:", round(accuracy_score(y_test, y_pred_rf), 3))
print("Precision:", round(precision_score(y_test, y_pred_rf), 3))
print("Recall:", round(recall_score(y_test, y_pred_rf), 3))
print("F1-poäng:", round(f1_score(y_test, y_pred_rf), 3))

print("\nKlassificeringsrapport:")
print(classification_report(y_test, y_pred_rf))

# Visa förväxlingsmatris
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    display_labels=["Gift/stabil", "Skild"]
)
plt.title("Förväxlingsmatris – Random Forest")
plt.show()

# --------------------------------------------------
# 3. Korsvalidering av Random Forest
# --------------------------------------------------

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate
)

# Samma korsvalidering som i Irfans modellering
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=RANDOM_STATE
)

# Utvärdera modellen på träningsdata
cv_resultat_rf = cross_validate(
    random_forest,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "f1", "roc_auc"]
)

# Visa medelvärde och standardavvikelse
print("\n=== Korsvalidering: Random Forest ===")
print(
    "Genomsnittlig accuracy:",
    round(cv_resultat_rf["test_accuracy"].mean(), 3)
)
print(
    "Genomsnittlig F1-poäng:",
    round(cv_resultat_rf["test_f1"].mean(), 3)
)
print(
    "Standardavvikelse F1:",
    round(cv_resultat_rf["test_f1"].std(), 3)
)
print(
    "Genomsnittlig ROC-AUC:",
    round(cv_resultat_rf["test_roc_auc"].mean(), 3)
)

# --------------------------------------------------
# 4. XGBoost
# --------------------------------------------------

from xgboost import XGBClassifier

# Skapa och träna XGBoost-modellen
xgboost_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    eval_metric="logloss"
)

xgboost_model.fit(X_train, y_train)

# Gör prediktioner på testdata
y_pred_xgb = xgboost_model.predict(X_test)

# Utvärdera modellen
print("\n=== XGBoost ===")
print("Träffsäkerhet:", round(accuracy_score(y_test, y_pred_xgb), 3))
print("Precision:", round(precision_score(y_test, y_pred_xgb), 3))
print("Recall:", round(recall_score(y_test, y_pred_xgb), 3))
print("F1-poäng:", round(f1_score(y_test, y_pred_xgb), 3))

print("\nKlassificeringsrapport:")
print(classification_report(y_test, y_pred_xgb))

# Korsvalidering med samma metod som Random Forest
cv_resultat_xgb = cross_validate(
    xgboost_model,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "f1", "roc_auc"]
)

print("\n=== Korsvalidering: XGBoost ===")
print("Genomsnittlig accuracy:", round(cv_resultat_xgb["test_accuracy"].mean(), 3))
print("Genomsnittlig F1-poäng:", round(cv_resultat_xgb["test_f1"].mean(), 3))
print("Standardavvikelse F1:", round(cv_resultat_xgb["test_f1"].std(), 3))
print("Genomsnittlig ROC-AUC:", round(cv_resultat_xgb["test_roc_auc"].mean(), 3))

# --------------------------------------------------
# 5. Förväxlingsmatris för XGBoost
# --------------------------------------------------

# Visa hur många observationer modellen klassificerar rätt och fel
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb,
    display_labels=["Gift/stabil", "Skild"],
    cmap="Blues"
)

plt.title("Förväxlingsmatris – XGBoost")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. SHAP-analys av XGBoost
# --------------------------------------------------

import shap

# Beräkna SHAP-värden för testdata
explainer_xgb = shap.TreeExplainer(xgboost_model)
shap_values_xgb = explainer_xgb(X_test)

# Visa vilka variabler som påverkar modellens prediktioner mest

shap.summary_plot(
    shap_values_xgb.values,
    X_test,
    show=False
)

plt.title("SHAP-analys – XGBoost")
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "reports" / "shap_xgboost.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 7. SHAP-analys av Random Forest
# --------------------------------------------------

# Beräkna SHAP-värden för Random Forest
explainer_rf = shap.TreeExplainer(random_forest)
shap_values_rf = explainer_rf(X_test)

# Välj SHAP-värden för klass 1 vid binär klassificering
rf_values = shap_values_rf.values

if rf_values.ndim == 3:
    rf_values = rf_values[:, :, 1]

# Visa de viktigaste variablerna
shap.summary_plot(
    rf_values,
    X_test,
    show=False
)

plt.title("SHAP-analys – Random Forest")
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "reports" / "shap_random_forest.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()
