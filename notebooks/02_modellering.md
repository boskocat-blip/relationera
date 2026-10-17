# Del 2: Modellering – baslinje och Decision Tree

**Ansvarig:** Irfan Pallani
**Projekt:** Identifiering av framgångsfaktorer för långsiktigt lyckliga relationer
**Kod:** `02_modellering_irfan.py`

---

## 1. Syfte

Vår grupp har valt att arbeta med skilsmässodata, och vi använder dataset nummer 1, *Split or Stay: Divorce Predictor* från Kaggle, som primärt dataset. I den här delen bygger jag vidare på Biljanas datainsamling och förberedelse. Jag tränar en baslinjemodell (Logistic Regression) och en Decision Tree för att svara på två frågor:

1. Hur bra kan vi klassificera om ett par är skilt eller inte utifrån svaren i enkäten?
2. Vilka variabler hänger starkast ihop med skilsmässa?

Random Forest, XGBoost och SHAP ingår inte i min del utan görs av andra i gruppen.

**Så ska resultaten läsas:** modellerna visar samband och klassificering. De visar inte vad som *orsakar* en skilsmässa.

## 2. Data

Datasetet har 170 par och 54 frågor, plus en målvariabel (0 = gift/stabil, 1 = skild/separerad). Svaren är heltal från 0 till 4. Filen innehåller inga namn eller id-kolumner, bara svaren och målvariabeln. Fördelningen är jämn: 86 par i klass 0 och 84 i klass 1. Det finns inga saknade värden men 20 dubblettrader. Dubbletterna har vi behållit, som i del 1, eftersom olika par kan ha svarat exakt likadant på en enkät med få svarsalternativ.

## 3. Upplägg

**Uppdelning först.** Jag delade upp datan i 80 % träning (136 par) och 20 % test (34 par) innan jag gjorde något annat. Delningen är stratifierad så att klasserna är lika fördelade i båda delarna, och testdatan rördes inte förrän modellerna var valda.

**Skalning och SMOTE bara på träningsdata.** Jag använde rådatan (`data/raw/divorce.csv`) och inte den skalade filen från del 1. Den filen är skalad på hela datasetet innan uppdelningen, och då "tittar" skalningen på testdatan. I stället ligger `StandardScaler` och `SMOTE` inne i en pipeline, så att de lär sig från träningsdelen i varje steg av korsvalideringen och aldrig från validerings- eller testdatan.

**Modeller:**

- Logistic Regression (baslinje), med skalade variabler.
- Decision Tree. Djup och minsta bladstorlek valdes med korsvalidering på träningsdatan (resultat: max djup 2, minst 3 par per blad).
- Båda modellerna körde jag också med SMOTE i träningsdelen.

**Utvärdering:** upprepad korsvalidering (5 delar, 5 gånger) på träningsdatan, och sedan en slutlig mätning på testdatan.

## 4. Resultat

| Modell | Korsvalidering F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0,984 | 0,941 | 1,000 | 0,882 | 0,938 |
| Logistic Regression + SMOTE | 0,984 | 0,941 | 1,000 | 0,882 | 0,938 |
| Decision Tree | 0,975 | 0,912 | 1,000 | 0,824 | 0,903 |
| Decision Tree + SMOTE | 0,974 | 0,912 | 1,000 | 0,824 | 0,903 |

Av de 34 testparen klassificerade Logistic Regression 32 rätt och Decision Tree 31 rätt. Båda missade bara par som faktiskt var skilda (2 respektive 3 stycken) och märkte aldrig ett stabilt par som skilt. Se `reports/irfan_modellering/forvaxlingsmatriser.png`.

**SMOTE:** Den gjorde ingen skillnad. Klasserna är redan jämnt fördelade (86 mot 84), så det fanns nästan inget att balansera. Jag lät SMOTE vara kvar i jämförelsen för att visa att den används på rätt sätt, bara på träningsdata, men den behövs inte för det här datasetet.

## 5. Samband

**Logistic Regression** (standardiserade variabler, så koefficienterna går att jämföra). De tio starkaste sambanden med skilsmässa:

| Variabel | Koefficient |
|---|---|
| No_home_time | 0,583 |
| good_to_leave_home | 0,531 |
| Aggro_argue | 0,444 |
| idk_what's_going_on | 0,441 |
| anxieties | 0,401 |
| happy | 0,366 |
| marriage | 0,363 |
| you're_inadequate | 0,360 |
| dreams | 0,359 |
| hate_subjects | 0,337 |

Alla tio är positiva, vilket betyder att högre svarsvärde hänger ihop med större sannolikhet för skilsmässa. Se `logreg_koefficienter.png`.

**Decision Tree:** trädet med djup 2 lägger nästan all vikt (0,99) på en enda variabel, `harmony`, och en liten del på `hate_subjects`. Se `decision_tree_betydelse.png`.

**Att tänka på vid tolkningen:** att trädet nästan bara använder `harmony` betyder inte att de andra variablerna är oviktiga. Många av frågorna mäter nästan samma sak och hänger ihop, så trädet kan hitta en enda fråga som räcker. Logistic Regression sprider vikten över många variabler. Därför ska man titta på båda och inte tolka rangordningen som exakt.

## 6. Begränsningar

- **Litet dataset.** Testdatan är bara 34 par, så ett enda fel ändrar träffsäkerheten med ungefär 3 procentenheter. Korsvalideringen är därför ett säkrare mått än testresultatet.
- **Dubbletter.** 5 av 34 testrader har en identisk rad i träningsdatan, vilket kan göra testet lite lättare. På de 29 testrader som inte har någon identisk träningsrad blev accuracy 0,931 för Logistic Regression och 0,897 för Decision Tree. Det är nästan samma som på hela testdatan, så dubbletterna förklarar inte de höga siffrorna.
- **Mycket höga resultat.** Båda modellerna ligger över 90 %, och korsvalideringens F1 är nära 0,98. Det tyder på att enkätsvaren skiljer de två grupperna väldigt tydligt, men vi vet inte hur det skulle fungera på andra par eller andra enkäter. Svaren kan också ha påverkats av hur relationen redan såg ut när enkäten besvarades.
- **Samband, inte orsak.** Se avsnitt 1.

## 7. Så upprepar du körningen

```bash
python notebooks/02_modellering_irfan.py
```

Slumptalet är fast (42), så resultaten blir likadana varje gång. Resultattabeller och bilder sparas i `reports/irfan_modellering/`. Biblioteken som behövs finns i `README_irfan.md`.
