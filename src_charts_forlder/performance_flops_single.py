import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso corretto della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../data")  # Percorso relativo alla posizione dello script

# Directory di destinazione per i grafici generati
output_dir = os.path.join(os.path.dirname(__file__), "../output_graphs")  # Modifica questa variabile come preferisci
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Crea la directory se non esiste

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

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

# Funzione per generare e salvare il grafico
def genera_grafico(titolo, computazioni_tipo, matrice_nome):
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

    # Salva il grafico in un file
    output_path = os.path.join(output_dir, f"{matrice_nome}_{titolo.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.savefig(output_path)
    plt.close()

# Genera e salva i grafici per ogni matrice e tipo di computazione
for file_name in os.listdir(dir_csv):
    if file_name.endswith(".csv"):
        # Estrai il nome della matrice (senza estensione .csv)
        matrice_nome = os.path.splitext(file_name)[0]

        # Filtra i dati per il file corrente (matrice)
        dati_matrice = dati[dati['nameMatrix'] == matrice_nome]

        # Genera i grafici per ciascun tipo di computazione
        genera_grafico(f"Prestazioni OpenMP {matrice_nome} (CSR e HLL) vs. Numero di Thread", computazioni_openmp, matrice_nome)
        genera_grafico(f"Prestazioni CUDA CSR {matrice_nome} vs. Numero di Thread", computazioni_cuda_csr, matrice_nome)
        genera_grafico(f"Prestazioni CUDA HLL {matrice_nome} vs. Numero di Thread", computazioni_cuda_hll, matrice_nome)

# Messaggio di completamento
print(f"I grafici sono stati salvati nella cartella '{output_dir}'.")
