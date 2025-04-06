import pandas as pd
import os

# Cartella dei CSV
input_folder = "../data"
output_file = "filtered_results.csv"

filtered_data = []

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        matrix_name = os.path.splitext(filename)[0]  # Rimuove ".csv"

        try:
            df = pd.read_csv(filepath)

            if "computation_type" in df.columns:
                # Filtra solo le righe "serial_csr"
                filtered_df = df[df["computation_type"] == "serial_hll"].copy()
                
                # Aggiungi colonna all'inizio
                filtered_df.insert(0, "name", matrix_name)

                filtered_data.append(filtered_df)
        except Exception as e:
            print(f"Errore nel file {filename}: {e}")

# Salva tutto in un unico CSV
if filtered_data:
    result_df = pd.concat(filtered_data)
    result_df.to_csv(output_file, index=False)
    print(f"File generato con successo: {output_file}")
else:
    print("Nessun dato 'serial_csr' trovato.")
