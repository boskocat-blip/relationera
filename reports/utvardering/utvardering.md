# Utvärdering av modeller

## 1. Syfte

Syftet med utvärderingen är att undersöka hur väl maskininlärningsmodellerna kan klassificera relationer som stabila eller skilda utifrån datasetets variabler. Vi jämför Logistic Regression och Decision Tree, både med och utan SMOTE.

## 2. Utvärderingsmetoder

Modellerna utvärderades med Accuracy, Precision, Recall, F1-score och ROC-AUC. Vi använde även korsvalidering och förväxlingsmatriser för att undersöka modellernas prestationer.

Datasetet innehåller 170 observationer och 54 förklarande variabler. Av dessa användes 136 observationer till träning och 34 till testning.

## 3. Resultat

| Modell | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 94,1 % | 93,8 % | 100 % |
| Logistic Regression + SMOTE | 94,1 % | 93,8 % | 100 % |
| Decision Tree | 91,2 % | 90,3 % | 93,1 % |
| Decision Tree + SMOTE | 91,2 % | 90,3 % | 93,1 % |

Resultaten visar att Logistic Regression uppnådde bäst resultat. Modellen hade högre accuracy och F1-score än Decision Tree.

SMOTE förbättrade inte resultaten för någon av modellerna.

## 4. Analys av förväxlingsmatriser

Logistic Regression klassificerade 32 av 34 observationer korrekt. Modellen identifierade samtliga 17 stabila relationer och 15 av 17 skilsmässor.

Decision Tree klassificerade 31 av 34 observationer korrekt. Den identifierade också samtliga stabila relationer, men endast 14 av 17 skilsmässor.

Logistic Regression gjorde alltså färre felklassificeringar.

## 5. Korsvalidering och överanpassning

Logistic Regression uppnådde 98,5 % accuracy vid korsvalidering och 94,1 % på testdatan. Skillnaden var 4,4 procentenheter.

Decision Tree uppnådde 97,5 % vid korsvalidering och 91,2 % på testdatan. Skillnaden var 6,3 procentenheter.

Båda modellerna presterade sämre på testdatan än vid korsvalideringen. Detta kan vara ett tecken på överanpassning, men kan också bero på variationer i ett litet dataset.

Logistic Regression hade en mindre skillnad och högre testresultat.

## 6. Kontroll av dubbletter

Vid kontroll av datasetet upptäcktes 20 dubblettrader. Dessutom hade 5 av de 34 testobservationerna identiska variabelvärden med minst en observation i träningsdatan.

För att undersöka detta utvärderades modellerna även på de 29 testobservationer som inte hade identiska träningsrader.

Logistic Regression uppnådde då 93,1 % accuracy och Decision Tree 89,7 %.

Resultaten visar att Logistic Regression fortfarande presterade bättre. Samtidigt visar kontrollen att dubbletter kan påverka resultatens tillförlitlighet.

Kontrollen tar endast bort överlappande observationer från själva utvärderingen. Den ersätter därför inte ett nytt experiment där dubbletter hanteras innan tränings- och testdata delas upp.

## 7. Slutsats

Utifrån utvärderingen rekommenderas Logistic Regression utan SMOTE som den mest lämpliga modellen av de testade alternativen.

Modellen uppnådde högst F1-score och accuracy, gjorde färre felklassificeringar och hade en mindre skillnad mellan korsvalidering och testresultat.

En begränsning är att datasetet är litet och innehåller dubbletter. Därför kan vi inte vara säkra på att modellen skulle prestera lika bra på helt nya observationer.

För att förbättra projektet rekommenderas framtida tester med ett större och mer varierat dataset samt en ny utvärdering där dubbletter hanteras innan datauppdelningen.