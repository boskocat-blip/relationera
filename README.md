# Identifiering av framgångsfaktorer för långsiktigt lyckliga relationer 
# Identifiering av framgångsfaktorer för långsiktigt lyckliga relationer

## Om projektet
Många relationer tar slut trots att paren från början verkar kompatibla. Målet med detta projekt är att använda dataanalys och maskininlärning för att identifiera vilka beteendemönster och faktorer som har störst betydelse för långsiktig relationstillfredsställelse.

Projektet fokuserar på mönsteridentifiering och faktoranalys för att hjälpa forskare, parterapeuter och individer att bättre förstå relationsdynamik.

## Gruppmedlemmar
* Nora Masamra
* Irfan Pallani
* Biljana Markovic
* Mathurin Radabud

## Datakällor
Projektet kombinerar och analyserar fyra komplementära dataset från Kaggle:
1. **Split or Stay: Divorce Predictor** (Huvuddataset baserat på Gottman DPS Skala)
2. **The Keys To Predict Divorce** (Valideringsdataset med Likert-skala 0-4)
3. **Social Media Addiction vs Relationship** (Analys av digital skärmtid och konflikter)
4. **Happy Couples Dataset** (Kontinuerlig skala för relationstillfredsställelse)

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
