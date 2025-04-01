import os
import glob
import pandas as pd

# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../../data")

# Lista di tutti i file CSV nella directory
data_files = glob.glob(os.path.join(dir_csv, "*.csv"))

# Lista per memorizzare i risultati
results = []

# Processa ogni file CSV
for file_path in data_files:
    df = pd.read_csv(file_path)
    name_matrix = os.path.basename(file_path)  # Nome del file come identificatore
    
    # Controlla se esiste la colonna "non_zero_values", altrimenti la imposta a None
    if "non_zero_values" in df.columns:
        df["non_zero_values"] = df["non_zero_values"].astype(int)  # Assicura il tipo intero
    else:
        df["non_zero_values"] = None  # Se non presente, assegna None

    # Estrai i tempi seriali come dizionario per accesso rapido
    serial_times = {
        row['computation_type']: row['time_used']
        for _, row in df.iterrows()
        if row['computation_type'] in ['serial_csr', 'serial_hll']
    }

    # Processa solo le righe parallele (OpenMP e CUDA)
    for _, row in df.iterrows():
        num_threads = row['threads_used']
        computation_type = row['computation_type']
        time_parallel = row['time_used']
        non_zero_values = row['non_zero_values']  # Recupera il valore dei non-zeri

        # Identifica la base (csr o hll) per ogni computation_type
        if "csr" in computation_type:
            base_type = "csr"
        elif "hll" in computation_type:
            base_type = "hll"
        else:
            continue  # Se non contiene né "csr" né "hll", viene ignorato

        # Cerca il tempo seriale corrispondente
        time_serial = serial_times.get(f"serial_{base_type}")

        # Se il tempo seriale esiste e il tempo parallelo è valido, calcola lo speedup
        if time_serial is not None and time_parallel > 0:
            speedup = time_serial / time_parallel
            results.append([name_matrix, num_threads, computation_type, non_zero_values, speedup])

# Creare il DataFrame con i risultati
output_df = pd.DataFrame(results, columns=["name_matrix", "num_threads", "computation_type", "non_zero_values", "speedup"])

# Salvare il CSV
output_csv_path = os.path.join(dir_csv, "speedup_single_csv/speedup_results.csv")
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # Crea la cartella se non esiste
output_df.to_csv(output_csv_path, index=False)

print(f"File di output generato: {output_csv_path}")
