from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# 1. HITTA PROJEKTMAPPEN


projektmapp = Path(__file__).resolve().parent.parent

# Leta efter modellernas resultat
resultatfiler = list(
    (projektmapp / "reports").rglob("resultat_modeller.csv")
)

if not resultatfiler:
    print("Ingen resultatfil hittades.")
    print("Kör modelleringen först för att skapa resultaten.")

else:

    
    # 2. LÄS IN RESULTATEN
    

    resultat = pd.read_csv(resultatfiler[0])

    print("\n=== UTVÄRDERING AV MODELLER ===\n")

    # Sortera efter högst F1-score
    resultat = resultat.sort_values(
        by="test_f1",
        ascending=False
    )

    print(resultat.to_string(index=False))

    
    # 3. HITTA BÄSTA MODELLEN
    

    basta_modell = resultat.iloc[0]

    print("\n=== BÄSTA MODELLEN ===")
    print("Modell:", basta_modell["modell"])
    print("Accuracy:", round(basta_modell["test_accuracy"], 3))
    print("Precision:", round(basta_modell["test_precision"], 3))
    print("Recall:", round(basta_modell["test_recall"], 3))
    print("F1-score:", round(basta_modell["test_f1"], 3))
    print("ROC-AUC:", round(basta_modell["test_roc_auc"], 3))

    
    # 4. SKAPA MAPP FÖR UTVÄRDERING
    

    output_mapp = projektmapp / "reports" / "utvardering"
    output_mapp.mkdir(parents=True, exist_ok=True)

    # Spara modellernas resultat
    resultat.to_csv(
        output_mapp / "modelljamforelse.csv",
        index=False
    )

    
    # 5. DIAGRAM: F1-SCORE
    

    plt.figure(figsize=(10, 5))

    plt.bar(
        resultat["modell"],
        resultat["test_f1"],
        color="steelblue"
    )

    plt.title("Jämförelse av modellernas F1-score")
    plt.xlabel("Modell")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    plt.savefig(
        output_mapp / "f1_jamforelse.png",
        dpi=300
    )

    plt.show()
    plt.close()

    
    # 6. DIAGRAM: ACCURACY
    

    plt.figure(figsize=(10, 5))

    plt.bar(
        resultat["modell"],
        resultat["test_accuracy"],
        color="seagreen"
    )

    plt.title("Jämförelse av modellernas Accuracy")
    plt.xlabel("Modell")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    plt.savefig(
        output_mapp / "accuracy_jamforelse.png",
        dpi=300
    )

    plt.show()
    plt.close()

    
    # 7. SKRIV UT SLUTSATS
    

    print("\n=== SLUTSATS ===")

    print(
        f"Modellen {basta_modell['modell']} "
        f"har högst F1-score på testdatan "
        f"({basta_modell['test_f1']:.3f})."
    )

    print(
        "Resultaten bör även granskas med "
        "hjälp av korsvalidering, confusion "
        "matrix och kontroll av överanpassning."
    )

    print("\nUtvärderingen är klar!")
    print("Resultaten har sparats i:", output_mapp)



    
    # 8. KORSVALIDERING OCH ÖVERANPASSNING
    

    print("\n=== KORSVALIDERING OCH TESTRESULTAT ===")

    # Beräkna skillnaden mellan korsvalidering och test
    resultat["skillnad_accuracy"] = (
        resultat["cv_accuracy"] - resultat["test_accuracy"]
    )

    # Visa resultaten
    print(
        resultat[
            [
                "modell",
                "cv_accuracy",
                "test_accuracy",
                "skillnad_accuracy"
            ]
        ].to_string(index=False)
    )

    # Spara jämförelsen
    resultat.to_csv(
        output_mapp / "korsvalidering_jamforelse.csv",
        index=False
    )

    print("\nKorsvalideringen har utvärderats!")
    
    
    # 9. KONTROLL AV DUBBLETTER
   

    print("\n=== KONTROLL AV DUBBLETTER ===")

    datafil = projektmapp / "data" / "raw" / "divorce.csv"

    data = pd.read_csv(datafil, sep=";")

    if data.shape[1] == 1:
        data = pd.read_csv(datafil, sep=",")

    data.columns = data.columns.str.strip()

    # Räkna dubbletter i datasetet
    antal_dubbletter = data.duplicated().sum()

    print("Antal dubbletter:", antal_dubbletter)

    # Samma uppdelning som i modelleringen
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Kontrollera identiska rader
    traningsrader = set(map(tuple, X_train.to_numpy()))

    kopior_i_test = sum(
        tuple(rad) in traningsrader
        for rad in X_test.to_numpy()
    )

    print("Antal observationer i testdata:", len(X_test))
    print("Identiska observationer i träningsdata:", kopior_i_test)

    if kopior_i_test > 0:
        print("VARNING: Dubbletter kan påverka testresultaten.")
    else:
        print("Inga identiska observationer mellan träning och test.")


