import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Sökväg till rådata
raw_data_path = '../data/raw/divorce.csv' if os.path.exists('../data/raw/divorce.csv') else 'data/raw/divorce.csv'

# Datainsamling och inläsning
try:
    df_raw = pd.read_csv(raw_data_path, sep=';')
    if df_raw.shape[1] == 1:
        df_raw = pd.read_csv(raw_data_path, sep=',')
except Exception as e:
    print(f"Fel vid inläsning av data: {e}")

# Rensa alla kolumnnamn från extra blanksteg
df_raw.columns = df_raw.columns.str.strip()

print("=== DATAINSAMLING OCH KVALITETSKONTROLL ===")
print(f"Antal observationer (rader): {df_raw.shape[0]}")
print(f"Antal variabler (kolumner): {df_raw.shape[1]}")

# Kontroll av saknade värden och dubbletter
missing = df_raw.isnull().sum().sum()
duplicates = df_raw.duplicated().sum()

print(f"Totalt antal saknade värden: {missing}")
print(f"Totalt antal dubbletter: {duplicates}")

# Målvariabeln är den sista kolumnen i datasetet
target_col = df_raw.columns[-1]

# Separering av funktioner (X) och målvariabel (y)
X = df_raw.iloc[:, :-1]
y = df_raw.iloc[:, -1]

# Standardisering av variabler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Skapar en behandlad DataFrame med skalade värden
df_processed = pd.DataFrame(X_scaled, columns=X.columns)
df_processed['Class'] = y.reset_index(drop=True)

# Säkerställer att mappen data/processed finns
processed_dir = '../data/processed' if os.path.exists('../data') else 'data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Sparar det förberedda datasetet
output_path = os.path.join(processed_dir, 'divorce_processed.csv')
df_processed.to_csv(output_path, index=False)

print(f"\n[OK] Datapreparering klar! Det rengjorda datasetet har sparats till: {output_path}")


