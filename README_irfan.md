# Del 2: Modellering (Irfan)

Den här delen tränar och utvärderar två modeller på datasetet *Split or Stay: Divorce Predictor* (dataset nummer 1, Kaggle): en baslinje med Logistic Regression och en Decision Tree. Den bygger vidare på del 1 (datainsamling och förberedelse).

## Filer

| Fil | Innehåll |
|---|---|
| `02_modellering_irfan.py` | Koden: uppdelning, modeller, korsvalidering, utvärdering och figurer |
| `02_modellering_irfan.md` | Texten: upplägg, resultat, samband och begränsningar |

## Installation

 Python version 3.12.

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

## Köra

Kör från projektmappen (där `data/` ligger):

```bash
python notebooks/02_modellering_irfan.py
```

Skriptet läser `data/raw/divorce.csv`, skriver resultaten i terminalen och sparar tabeller och bilder i `reports/irfan_modellering/`. Slumptalet är fast, så två körningar ger samma resultat.

## Vad som skapas

- `resultat_modeller.csv` - korsvalidering och testresultat för alla fyra modeller
- `logreg_koefficienter.csv` och `.png` - sambanden i Logistic Regression
- `decision_tree_betydelse.csv` och `.png` - variabelbetydelse i Decision Tree
- `forvaxlingsmatriser.png` - förväxlingsmatriser på testdatan

## Viktigt att veta

- Datan delas upp i träning och test **innan** något skalas. `StandardScaler` och `SMOTE` ligger i en pipeline och används därför bara på träningsdata.
- Skriptet använder rådatan (`data/raw/divorce.csv`) och inte den skalade filen i `data/processed/`, eftersom den är skalad på hela datasetet före uppdelningen.