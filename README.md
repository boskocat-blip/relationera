# Identifiering av framgångsfaktorer för långsiktigt lyckliga relationer

## Om projektet
Många relationer tar slut trots att paren från början verkar kompatibla. Målet med detta projekt är att använda dataanalys och maskininlärning för att identifiera vilka beteendemönster och faktorer som har störst betydelse för långsiktig relationstillfredsställelse.

Projektet fokuserar på mönsteridentifiering och faktoranalys för att hjälpa forskare, parterapeuter och individer att bättre förstå relationsdynamik.

## Gruppmedlemmar
* Nora Masamra
* Irfan Pallani
* Biljana Markovic
* Mathurin Radabud
* Abdullahi Sheek Elmi

## Datakällor

Projektet kombinerar och analyserar komplementär dataset från Kaggle:

**Split or Stay: Divorce Predictor** (Huvuddataset baserat på Gottman DPS Skala).

Vi tänker att välja **skildmässa** data, och nummer och dataset **Split or Stay: Divorce Predictor.**


## Projektstruktur:

├── data/
│   ├── raw/             # Ursprungliga datafiler (t.ex. divorce.csv)
│   └── processed/       # Rensade och harmoniserade dataset
├── notebooks/           # Jupyter Notebooks för EDA och modeller
├── src/                 # Källkod och hjälpfunktioner
├── .gitignore           # Git-exkluderingsregler
└── README.md            # Projektbeskrivning


## Metodkombination och Modeller
* **Baseline:** Logistic Regression
* **Avancerade modeller:** Decision Tree, Random Forest och XGBoost
* **Förklarbarhet (XAI):** SHAP-värden för identifiering av viktiga faktorer
| Fil | Innehåll |
|---|---|
| `02_modellering.py` | Koden: uppdelning, modeller, korsvalidering, utvärdering och figurer |
| `02_modellering.md` | Texten: upplägg, resultat, samband och begränsningar |

## Installation

 Python version 3.12.

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

## Köra

Kör från projektmappen (där `data/` ligger):

```bash
python notebooks/02_modellering.py
```

Skriptet läser `data/raw/divorce.csv`, skriver resultaten i terminalen och sparar tabeller och bilder. Slumptalet är fast, så två körningar ger samma resultat.

## Vad som skapas

- `resultat_modeller.csv` - korsvalidering och testresultat för alla fyra modeller
- `logreg_koefficienter.csv` och `.png` - sambanden i Logistic Regression
- `decision_tree_betydelse.csv` och `.png` - variabelbetydelse i Decision Tree
- `forvaxlingsmatriser.png` - förväxlingsmatriser på testdatan

## Viktigt att veta

- Datan delas upp i träning och test **innan** något skalas. `StandardScaler` och `SMOTE` ligger i en pipeline och används därför bara på träningsdata.
- Skriptet använder rådatan (`data/raw/divorce.csv`) och inte den skalade filen i `data/processed/`, eftersom den är skalad på hela datasetet före uppdelningen.