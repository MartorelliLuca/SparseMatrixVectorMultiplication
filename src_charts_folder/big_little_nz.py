import os
import glob
import pandas as pd

# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../data")

# Percorso del file di output
output_file = os.path.join(os.path.dirname(__file__), "matrici_non_zeri_ordinati.txt")

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

# Dizionario per memorizzare i nomi delle matrici e il numero di valori non zero
matrici_non_zeri = {}

# Leggi tutti i file CSV e calcola il numero di valori non zero per ciascuna matrice
for file_name in os.listdir(dir_csv):
    if file_name.endswith(".csv"):
        file_path = os.path.join(dir_csv, file_name)
        df = pd.read_csv(file_path)

        # Verifica che la colonna "non_zero_values" esista
        if 'non_zero_values' not in df.columns:
            print(f"Attenzione: la colonna 'non_zero_values' non è presente nel file {file_name}. Ignorato.")
            continue

        # Estrai il numero totale di valori non zero dalla colonna "non_zero_values"
        non_zeri = df['non_zero_values'].sum()  # Somma tutti i valori della colonna

        matrici_non_zeri[file_name] = non_zeri

        # Diagnostica: Stampa il numero di valori non zero per ogni matrice
        print(f"File {file_name} ha {non_zeri} valori non zero nella colonna 'non_zero_values'.")

# Ordina le matrici in base al numero di valori non zero (dal più grande al più piccolo)
matrici_ordinate = sorted(matrici_non_zeri.items(), key=lambda x: x[1], reverse=True)

# Salva il risultato in un file .txt
with open(output_file, 'w') as f:
    for matrice, non_zeri in matrici_ordinate:
        f.write(f"{matrice}: {non_zeri}\n")

print(f"I nomi delle matrici sono stati salvati in {output_file}")
