import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso corretto della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../data")  # Percorso relativo alla posizione dello script

# Percorso della directory di output per i grafici
output_dir = os.path.join(os.path.dirname(__file__), "../charts/performance/mean_performance_plot")
os.makedirs(output_dir, exist_ok=True)  # Creazione della directory se non esiste

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

# Leggi tutti i file CSV nella directory
dati = []
for file_name in os.listdir(dir_csv):
    if file_name.endswith(".csv"):
        file_path = os.path.join(dir_csv, file_name)
        df = pd.read_csv(file_path)

        # Verifica che le colonne necessarie esistano nel file CSV
        if "threads_used" not in df.columns or "computation_type" not in df.columns or "GFLOPS" not in df.columns:
            print(f"Attenzione: Il file '{file_name}' manca di una colonna richiesta, ignorato.")
            continue  # Salta il file se mancano colonne

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

# Funzione per generare e salvare il grafico
def genera_grafico(titolo, computazioni_tipo, file_name):
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Salva il grafico invece di mostrarlo
    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path)
    plt.close()

    print(f"Grafico salvato: {plot_path}")

# Genera e salva i grafici
genera_grafico("Prestazioni OpenMP (CSR e HLL) vs. Numero di Thread", computazioni_openmp, "openmp_performance.png")
genera_grafico("Prestazioni CUDA CSR vs. Numero di Thread", computazioni_cuda_csr, "cuda_csr_performance.png")
genera_grafico("Prestazioni CUDA HLL vs. Numero di Thread", computazioni_cuda_hll, "cuda_hll_performance.png")