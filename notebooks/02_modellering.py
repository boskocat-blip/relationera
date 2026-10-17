# Del 2: Modellering (Irfan Pallani)
# Projekt: Identifiering av framgångsfaktorer för långsiktigt lyckliga relationer
# Vår grupp har valt att arbeta med skilsmässodata. Vi använder dataset nummer 1,
# "Split or Stay: Divorce Predictor" från Kaggle, som primärt dataset.
# Jag fortsätter på Biljanas (datainsamling och förberedelse).
# Jag använder samma data (data/raw/divorce.csv) och samma målvariabel
# (sista kolumnen: 0 = gift/stabil, 1 = skild/separerad).
#
# Jag gör bara min egen del: en baslinjemodell (Logistic Regression) och en
# Decision Tree. Random Forest, XGBoost och SHAP görs av andra i gruppen.
#
# Viktigt om tolkningen: modellerna visar SAMBAND och klassificerar par som
# skilda eller inte. De visar inte vad som orsakar en skilsmässa.
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sparar bilder till fil, ingen fönstervisning behövs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SLUMPTAL = 42  # samma slumptal överallt så att resultaten blir likadana varje gång

PROJEKTMAPP = Path(__file__).resolve().parents[1]
DATAFIL = PROJEKTMAPP / "data" / "raw" / "divorce.csv"
UTMAPP = PROJEKTMAPP / "reports" / "modellering"


# --------------------------------------------------------------------------
# 1. Läs in data
# --------------------------------------------------------------------------
def las_data(sokvag):
    """Läser divorce.csv. Filen kan ha semikolon eller komma som skiljetecken."""
    data = pd.read_csv(sokvag, sep=";")
    if data.shape[1] == 1:
        data = pd.read_csv(sokvag, sep=",")
    data.columns = data.columns.str.strip()
    return data


# --------------------------------------------------------------------------
# 2. Modellerna
# --------------------------------------------------------------------------
# SMOTE ligger inne i pipelinen. Då körs den bara på träningsdelen i varje
# korsvalidering och aldrig på valideringsdelen eller testdatan.
# StandardScaler ligger också i pipelinen av samma skäl: skalningen lär sig
# medelvärde och standardavvikelse bara från träningsdata.
def bygg_modeller():
    logreg = Pipeline(
        [
            ("skala", StandardScaler()),
            ("modell", LogisticRegression(max_iter=1000, random_state=SLUMPTAL)),
        ]
    )
    logreg_smote = Pipeline(
        [
            ("skala", StandardScaler()),
            ("smote", SMOTE(random_state=SLUMPTAL)),
            ("modell", LogisticRegression(max_iter=1000, random_state=SLUMPTAL)),
        ]
    )
    trad = Pipeline(
        [("modell", DecisionTreeClassifier(random_state=SLUMPTAL))]
    )
    trad_smote = Pipeline(
        [
            ("smote", SMOTE(random_state=SLUMPTAL)),
            ("modell", DecisionTreeClassifier(random_state=SLUMPTAL)),
        ]
    )

    # Trädet blir lätt för djupt och lär sig utantill, så vi väljer djup och
    # minsta bladstorlek med korsvalidering på träningsdata.
    parametrar = {
        "modell__max_depth": [2, 3, 4, 5, None],
        "modell__min_samples_leaf": [1, 3, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SLUMPTAL)
    return {
        "Logistic Regression": logreg,
        "Logistic Regression + SMOTE": logreg_smote,
        "Decision Tree": GridSearchCV(trad, parametrar, cv=cv, scoring="f1"),
        "Decision Tree + SMOTE": GridSearchCV(trad_smote, parametrar, cv=cv, scoring="f1"),
    }


def korsvalidera(modell, X_train, y_train):
    """Upprepad korsvalidering på träningsdata. Ger medelvärde och spridning."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SLUMPTAL)
    resultat = cross_validate(
        modell, X_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"]
    )
    return {
        "cv_accuracy": resultat["test_accuracy"].mean(),
        "cv_f1": resultat["test_f1"].mean(),
        "cv_f1_std": resultat["test_f1"].std(),
        "cv_roc_auc": resultat["test_roc_auc"].mean(),
    }


def utvardera(modell, X_test, y_test):
    """Mäter modellen på testdatan, som inte använts vid träningen."""
    gissning = modell.predict(X_test)
    sannolikhet = modell.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, gissning),
        "test_precision": precision_score(y_test, gissning),
        "test_recall": recall_score(y_test, gissning),
        "test_f1": f1_score(y_test, gissning),
        "test_roc_auc": roc_auc_score(y_test, sannolikhet),
    }


# --------------------------------------------------------------------------
# 3. Hjälpfunktioner för kontroller och figurer
# --------------------------------------------------------------------------
def antal_testrader_som_finns_i_traning(X_train, X_test):
    """Datan har dubbletter. Räknar testrader som är identiska med en träningsrad."""
    traningsrader = set(map(tuple, X_train.to_numpy()))
    return int(sum(tuple(rad) in traningsrader for rad in X_test.to_numpy()))


def traffsakerhet_pa_okanda_rader(modell, X_train, X_test, y_test):
    """Accuracy bara på testrader som inte har en identisk rad i träningsdatan."""
    traningsrader = set(map(tuple, X_train.to_numpy()))
    okanda = np.array([tuple(rad) not in traningsrader for rad in X_test.to_numpy()])
    gissning = modell.predict(X_test[okanda])
    return int(okanda.sum()), accuracy_score(y_test[okanda], gissning)


def spara_viktigaste_variabler(tabell, kolumn, rubrik, filnamn, antal=10):
    """Stapeldiagram över de variabler som modellen lägger mest vikt på."""
    top = tabell.reindex(tabell[kolumn].abs().sort_values(ascending=False).index).head(antal)
    top = top.iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["variabel"], top[kolumn])
    plt.title(rubrik)
    plt.tight_layout()
    plt.savefig(UTMAPP / filnamn, dpi=150)
    plt.close()


def main():
    UTMAPP.mkdir(parents=True, exist_ok=True)

    # ----- Läs in och kontrollera -----
    data = las_data(DATAFIL)
    malkolumn = data.columns[-1]  # sista kolumnen är målvariabeln
    X = data.drop(columns=malkolumn)
    y = data[malkolumn]

    print("=== DATA ===")
    print(f"Rader: {len(data)}, förklarande variabler: {X.shape[1]}, målvariabel: {malkolumn}")
    print("Fördelning i målvariabeln:")
    print(y.value_counts().to_string())
    print(f"Dubblettrader: {data.duplicated().sum()}  (behålls, som i del 1)")

    # ----- Dela upp INNAN något skalas eller balanseras -----
    # Testdatan lämnas orörd tills modellerna är valda.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SLUMPTAL
    )
    print(f"\nTräning: {len(X_train)} rader, test: {len(X_test)} rader")
    delar = antal_testrader_som_finns_i_traning(X_train, X_test)
    print(f"Testrader som har en identisk rad i träningsdatan: {delar} av {len(X_test)}")

    # ----- Träna och utvärdera -----
    modeller = bygg_modeller()
    rader = []
    tranade = {}
    for namn, modell in modeller.items():
        print(f"\n=== {namn} ===")
        cv_resultat = korsvalidera(modell, X_train, y_train)
        modell.fit(X_train, y_train)
        if isinstance(modell, GridSearchCV):
            print("Valda inställningar:", modell.best_params_)
        test_resultat = utvardera(modell, X_test, y_test)
        tranade[namn] = modell
        rader.append({"modell": namn, **cv_resultat, **test_resultat})
        print(
            f"Korsvalidering F1: {cv_resultat['cv_f1']:.3f} (std {cv_resultat['cv_f1_std']:.3f}), "
            f"test F1: {test_resultat['test_f1']:.3f}, test accuracy: {test_resultat['test_accuracy']:.3f}"
        )

    resultat = pd.DataFrame(rader).round(3)
    resultat.to_csv(UTMAPP / "resultat_modeller.csv", index=False)
    print("\n=== SAMMANSTÄLLNING ===")
    print(resultat.to_string(index=False))

    # Dubbletter kan göra att testet ser lättare ut än det är. Därför kontrolleras
    # även träffsäkerheten på testrader som modellen inte har sett en kopia av.
    print("\n=== KONTROLL: BARA TESTRADER UTAN IDENTISK TRÄNINGSRAD ===")
    for namn in ["Logistic Regression", "Decision Tree"]:
        antal, traff = traffsakerhet_pa_okanda_rader(tranade[namn], X_train, X_test, y_test)
        print(f"{namn}: accuracy {traff:.3f} på {antal} rader")

    # ----- Förväxlingsmatriser för modellerna utan SMOTE -----
    fig, axar = plt.subplots(1, 2, figsize=(9, 4))
    for ax, namn in zip(axar, ["Logistic Regression", "Decision Tree"]):
        ConfusionMatrixDisplay.from_estimator(
            tranade[namn], X_test, y_test, ax=ax, colorbar=False,
            display_labels=["Gift/stabil", "Skild"],
        )
        ax.set_title(namn)
    plt.tight_layout()
    plt.savefig(UTMAPP / "forvaxlingsmatriser.png", dpi=150)
    plt.close()

    # ----- Samband: vilka variabler hänger ihop med skilsmässa? -----
    logreg = tranade["Logistic Regression"].named_steps["modell"]
    koefficienter = pd.DataFrame({"variabel": X.columns, "koefficient": logreg.coef_[0]})
    koefficienter = koefficienter.reindex(
        koefficienter["koefficient"].abs().sort_values(ascending=False).index
    )
    koefficienter.to_csv(UTMAPP / "logreg_koefficienter.csv", index=False)
    spara_viktigaste_variabler(
        koefficienter, "koefficient",
        "Logistic Regression: de 10 starkaste sambanden (standardiserade variabler)",
        "logreg_koefficienter.png",
    )

    trad = tranade["Decision Tree"].best_estimator_.named_steps["modell"]
    betydelse = pd.DataFrame({"variabel": X.columns, "betydelse": trad.feature_importances_})
    betydelse = betydelse.sort_values("betydelse", ascending=False)
    betydelse.to_csv(UTMAPP / "decision_tree_betydelse.csv", index=False)
    spara_viktigaste_variabler(
        betydelse, "betydelse", "Decision Tree: de 10 viktigaste variablerna",
        "decision_tree_betydelse.png",
    )

    print("\n=== 10 starkaste sambanden i Logistic Regression ===")
    print(koefficienter.head(10).round(3).to_string(index=False))
    print("\n=== 10 viktigaste variablerna i Decision Tree ===")
    print(betydelse.head(10).round(3).to_string(index=False))
    print(f"\nResultat och bilder sparade i {UTMAPP}")


if __name__ == "__main__":
    main()
