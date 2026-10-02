# Datainsamling och Dataförberedelse

**Ansvarig:** Biljana Markovic  
**Projekt:** Identifiering av framgångsfaktorer för långsiktigt lyckliga relationer  
**Datum:** Oktober 2026  

---

## 1. Syfte och Översikt
Detta dokument sammanfattar datainsamlingen och dataprepareringen för huvuddatasetet **Split or Stay: Divorce Predictor** (baserat på Gottman DPS Skala). Målet med denna fas är att säkerställa hög datakvalitet, hantera eventuella saknade värden samt standardisera alla variabler inför kommande modelleringssteg.

---

## 2. Resultat av Kvalitetskontroll (Data Audit)

Skriptet `01_datainsamling_och_forberedelse.py` kördes framgångsrikt och genererade följande resultat:

* **Antal observationer (rader):** 170 par
* **Antal variabler (kolumner):** 55 (54 oberoende frågor/prediktorer samt 1 målvariabel `Class`)
* **Saknade värden (Missing Values):** 0
* **Identifierade dubbletter:** 20 observationer

### Analys av dubbletter:
Datamängden innehåller 20 dubblettrader. Detta beror på att enkäten använder diskreta svarsskalor där olika par har angett identiska svarsprofiler över de 54 frågorna. Eftersom detta representerar genuina svar från populationen och inte felaktiga inmatningar behålls alla rader för att bevara urvalsstorleken ($N=170$).

---

## 3. Dataförberedelse och Transformationer

Följande transformeringssteg har genomförts på datamängden:

1. **Separering av variabler:** Målvariabeln (`Class`: 0 = Gift/Stabil, 1 = Skild/Separerad) separerades från de 54 prediktorfrågorna ($X$).
2. **Feature Scaling (StandardScaler):** Samtliga 54 oberoende variabler standardiserades med Z-score transformation:
   $$z = \frac{x - \mu}{\sigma}$$
   Detta säkerställer att alla funktioner har ett medelvärde på 0 och en standardavvikelse på 1, vilket förhindrar att variabler med större skala dominerar under modellträningen.
3. **Eksport:** Det färdigbehandlade och harmoniserade datasetet sparades till mappen `data/processed/divorce_processed.csv`.

---

## 4. Slutsats och Nästa Steg
Datainsamlingen och dataförberedelsen är 100% färdigställd. Datasetet är nu helt redo för vidare explorativ dataanalys (EDA), baseline-modellering (Logistic Regression) samt trädbaserade modeller (Random Forest och XGBoost).