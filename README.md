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
