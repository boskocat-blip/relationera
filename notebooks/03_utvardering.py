
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ==========================================
# 1. HITTA PROJEKTMAPPEN
# ==========================================

projektmapp = Path(__file__).resolve().parent.parent

output_mapp = projektmapp / "reports" / "utvardering"
output_mapp.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LAS IN MODELLRESULTAT
# ==========================================

# Resultat fran Logistic Regression och Decision Tree
resultatfiler = list(
    (projektmapp / "reports").rglob("resultat_modeller.csv")
)

if not resultatfiler:
    raise FileNotFoundError(
        "Ingen resultat_modeller.csv hittades. "
        "Kor 02_modellering.py forst."
    )

resultat = pd.read_csv(resultatfiler[0])

# ==========================================
# 3. LAGG TILL RANDOM FOREST OCH XGBOOST
# ==========================================

# Varden fran gruppens rapport:
# notebooks/02_random_forest_xgboost.md
# Dessa uppdateras manuellt om modellerna tranats om.

nya_resultat = pd.DataFrame([
    {
        "modell": "Random Forest",
        "cv_accuracy": 0.985,
        "cv_f1": 0.984,
        "cv_f1_std": 0.023,
        "cv_roc_auc": 0.999,
        "test_accuracy": 0.941,
        "test_precision": 1.000,
        "test_recall": 0.882,
        "test_f1": 0.938,
        "test_roc_auc": float("nan")
    },
    {
        "modell": "XGBoost",
        "cv_accuracy": 0.982,
        "cv_f1": 0.982,
        "cv_f1_std": 0.026,
        "cv_roc_auc": 0.984,
        "test_accuracy": 0.941,
        "test_precision": 1.000,
        "test_recall": 0.882,
        "test_f1": 0.938,
        "test_roc_auc": float("nan")
    }
])

# Undvik att samma modell laggs till tva ganger
resultat = resultat[
    ~resultat["modell"].isin(nya_resultat["modell"])
]

resultat = pd.concat(
    [resultat, nya_resultat],
    ignore_index=True
)


# ==========================================
# 4. JAMFOR ALLA MODELLER
# ==========================================

resultat = resultat.sort_values(
    by="test_f1",
    ascending=False
).reset_index(drop=True)

print("\n=== UTVARDERING AV MODELLER ===\n")

print(resultat.to_string(index=False))

resultat.to_csv(
    output_mapp / "modelljamforelse.csv",
    index=False
)


# ==========================================
# 5. HITTA MODELLER MED HOGST F1-SCORE
# ==========================================

basta_f1 = resultat["test_f1"].max()

basta_modeller = resultat[
    (resultat["test_f1"] - basta_f1).abs() < 0.000001
]

print("\n=== MODELLER MED HOGST F1-SCORE ===")

print(
    basta_modeller[
        ["modell", "test_accuracy", "test_f1"]
    ].to_string(index=False)
)

print("\nHogsta F1-score:", round(basta_f1, 3))


# ==========================================
# 6. DIAGRAM: F1-SCORE
# ==========================================

diagramdata = resultat.sort_values("test_f1")

plt.figure(figsize=(11, 6))

plt.barh(
    diagramdata["modell"],
    diagramdata["test_f1"],
    color="steelblue"
)

plt.title("Jamforelse av modellernas F1-score")
plt.xlabel("F1-score")
plt.ylabel("Modell")
plt.xlim(0, 1.05)

plt.tight_layout()

plt.savefig(
    output_mapp / "f1_jamforelse.png",
    dpi=300
)

plt.close()

print("\nF1-score-diagram sparat!")


# ==========================================
# 7. DIAGRAM: ACCURACY
# ==========================================

diagramdata = resultat.sort_values("test_accuracy")

plt.figure(figsize=(11, 6))

plt.barh(
    diagramdata["modell"],
    diagramdata["test_accuracy"],
    color="seagreen"
)

plt.title("Jamforelse av modellernas Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Modell")
plt.xlim(0, 1.05)

plt.tight_layout()

plt.savefig(
    output_mapp / "accuracy_jamforelse.png",
    dpi=300
)

plt.close()

print("Accuracy-diagram sparat!")


# ==========================================
# 8. KORSVALIDERING OCH OVERANPASSNING
# ==========================================

print("\n=== KORSVALIDERING OCH TESTRESULTAT ===")

resultat["skillnad_accuracy"] = (
    resultat["cv_accuracy"] -
    resultat["test_accuracy"]
)

kolumner = [
    "modell",
    "cv_accuracy",
    "test_accuracy",
    "skillnad_accuracy"
]

print(resultat[kolumner].to_string(index=False))

resultat[kolumner].to_csv(
    output_mapp / "korsvalidering_jamforelse.csv",
    index=False
)

print("\nKorsvalideringen har utvarderats!")


# ==========================================
# 9. KONTROLL AV DUBBLETTER
# ==========================================

print("\n=== KONTROLL AV DUBBLETTER ===")

datafil = projektmapp / "data" / "raw" / "divorce.csv"

if not datafil.exists():
    raise FileNotFoundError(
        f"Datasetet hittades inte: {datafil}"
    )

data = pd.read_csv(datafil, sep=";")

if data.shape[1] == 1:
    data = pd.read_csv(datafil, sep=",")

data.columns = data.columns.str.strip()

# Rakna dubbletter i hela datasetet
antal_dubbletter = data.duplicated().sum()

print("Antal observationer:", len(data))
print("Antal dubbletter:", antal_dubbletter)

# Separera forklarande variabler och malvariabel
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Samma uppdelning som gruppens modellering
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Kontrollera identiska observationer
traningsrader = set(
    map(tuple, X_train.to_numpy())
)

kopior_i_test = sum(
    tuple(rad) in traningsrader
    for rad in X_test.to_numpy()
)

print("Antal traningsobservationer:", len(X_train))
print("Antal testobservationer:", len(X_test))

print(
    "Testrader med identiska variabelvarden "
    "i traningsdata:",
    kopior_i_test
)

if kopior_i_test > 0:
    print(
        "VARNING: Identiska observationer kan "
        "paverka modellernas testresultat."
    )
else:
    print(
        "Inga identiska variabelrader hittades "
        "mellan traning och test."
    )


# ==========================================
# 10. ANALYS AV RESULTATEN
# ==========================================

print("\n=== ANALYS AV MODELLERNA ===")

print(
    "\nLogistic Regression, Random Forest och "
    "XGBoost har samma avrundade testresultat."
)

print(
    "Decision Tree har nagot lagre Accuracy "
    "och F1-score."
)

print(
    "SMOTE gav ingen forbattring av "
    "testresultaten for Logistic Regression "
    "eller Decision Tree."
)

print(
    "Skillnaderna mellan korsvalidering och "
    "testresultat kan bero pa overanpassning "
    "eller variation i det lilla datasetet."
)

print(
    "Dubbletter mellan traning och test "
    "innebar en risk for missvisande resultat."
)

print(
    "Kontroll av accuracy utan identiska "
    "traningsrader har tidigare gjorts for "
    "Logistic Regression och Decision Tree, "
    "men inte for Random Forest och XGBoost."
)


# ==========================================
# 11. SLUTSATS
# ==========================================

print("\n=== SLUTSATS ===")

print(
    "Logistic Regression, Random Forest och "
    "XGBoost uppnadde F1-score 0.938 och "
    "Accuracy 0.941 pa testdatan."
)

print(
    "Ingen av dessa tre modelltyper kan "
    "utses till ensam bast utifran dessa matt."
)

print(
    "Logistic Regression utan SMOTE "
    "rekommenderas som ett praktiskt modellval "
    "eftersom modellen ar relativt enkel "
    "att tolka."
)

print(
    "Resultaten bor tolkas forsiktigt eftersom "
    "datasetet ar litet och innehaller dubbletter."
)

print(
    "Ytterligare testning pa oberoende data "
    "rekommenderas."
)


# ==========================================
# 12. FARDIG
# ==========================================

print("\n====================================")
print("UTVARDERINGEN AR KLAR!")
print("====================================")

print("\nResultaten finns i:")
print(output_mapp)

print("\nFiler som har skapats:")
print("- modelljamforelse.csv")
print("- korsvalidering_jamforelse.csv")
print("- f1_jamforelse.png")
print("- accuracy_jamforelse.png")



