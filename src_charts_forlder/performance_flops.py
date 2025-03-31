import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso corretto della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../data")  # Percorso relativo alla posizione dello script

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

# Soglie per suddividere i grafici in base al numero di nonzeri
soglie_nonzeri = [10000, 100000, 500000, 1e6, 2.5e6, 1e7]

# Leggi tutti i file CSV nella directory
dati = []
for file_name in os.listdir(dir_csv):
    if file_name.endswith(".csv"):
        file_path = os.path.join(dir_csv, file_name)
        df = pd.read_csv(file_path)

        # Verifica che la colonna "threads_used" esista nel file CSV
        if "threads_used" not in df.columns:
            print(f"Attenzione: Il file '{file_name}' non contiene la colonna 'threads_used', ignorato.")
            continue  # Salta il file se manca la colonna

        # Verifica che la colonna "computation_type" esista nel file CSV
        if "computation_type" not in df.columns:
            print(f"Attenzione: Il file '{file_name}' non contiene la colonna 'computation_type', ignorato.")
            continue  # Salta il file se manca la colonna

        df["threads_used"] = df["threads_used"].astype(int)  # Assicurati che sia un intero
        dati.append(df)


# Concatenazione di tutti i dati in un unico DataFrame
if dati:
    dati = pd.concat(dati, ignore_index=True)
else:
    raise ValueError("Nessun file CSV trovato nella cartella 'data'.")

# Suddividi i tipi di computazione in 3 gruppi
computazioni_openmp = ['parallel_open_mp_csr', 'parallel_open_mp_hll']
computazioni_cuda_csr = ['cuda_csr_kernel_1', 'cuda_csr_kernel_2', 'cuda_csr_kernel_3', 'cuda_csr_kernel_4']
computazioni_cuda_hll = ['cuda_hll_kernel_1', 'cuda_hll_kernel_2', 'cuda_hll_kernel_3', 'cuda_hll_kernel_4']

# Funzione per generare il grafico
def genera_grafico(titolo, computazioni_tipo):
    plt.figure(figsize=(10, 6))
    
    for tipo in computazioni_tipo:
        valori = dati[dati['computation_type'] == tipo]

        # Raggruppa i dati per numero di thread
        thread_values = sorted(set(valori['threads_used']))
        average_performance = {t: [] for t in thread_values}
        
        for num_thread in thread_values:
            performance = valori[valori['threads_used'] == num_thread]['GFLOPS']
            average_performance[num_thread] = np.mean(performance) if len(performance) > 0 else 0

        # Traccia la linea per il tipo di computazione
        plt.plot(list(average_performance.keys()), list(average_performance.values()), label=f"{tipo}")
    
    # Impostazioni del grafico
    plt.title(titolo)
    plt.xlabel("Numero di Thread")
    plt.ylabel("Prestazioni (GFLOPS)")

    # Posiziona la legenda fuori dal grafico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

# Genera i grafici
genera_grafico("Prestazioni OpenMP (CSR e HLL) vs. Numero di Thread", computazioni_openmp)
genera_grafico("Prestazioni CUDA CSR vs. Numero di Thread", computazioni_cuda_csr)
genera_grafico("Prestazioni CUDA HLL vs. Numero di Thread", computazioni_cuda_hll)

# Mostra i grafici
plt.show()
