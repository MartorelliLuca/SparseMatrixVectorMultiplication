import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../../data")

# Percorso della directory di output per i grafici
output_dir = os.path.join(os.path.dirname(__file__), "../charts/performance/individual_performance_plots")
os.makedirs(output_dir, exist_ok=True)

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

# Leggi tutti i file CSV nella directory
dati = []
for file_name in os.listdir(dir_csv):
    if file_name.endswith(".csv"):
        file_path = os.path.join(dir_csv, file_name)
        df = pd.read_csv(file_path)

        if "threads_used" not in df.columns or "computation_type" not in df.columns or "GFLOPS" not in df.columns:
            print(f"Attenzione: Il file '{file_name}' manca di una colonna richiesta, ignorato.")
            continue

        df["threads_used"] = df["threads_used"].astype(int)
        df["file_name"] = file_name  # Aggiungi il nome del file per l'identificazione nella legenda
        dati.append(df)

# Concatenazione di tutti i dati in un unico DataFrame
if dati:
    dati = pd.concat(dati, ignore_index=True)
else:
    raise ValueError("Nessun file CSV trovato nella cartella 'data'.")

# Lista di tutte le computazioni presenti nei dati
tipi_computazione = dati['computation_type'].unique()

# Kernel che richiedono la moltiplicazione dei thread per 32
kernel_moltiplica_32 = ['cuda_csr_kernel_2', 'cuda_csr_kernel_3', 'cuda_csr_kernel_4']

# Funzione per generare e salvare il grafico per ogni configurazione
def genera_grafico_kernel(computation_type):
    plt.figure(figsize=(12, 6))  # Ingrandito il grafico (larghezza aumentata)

    valori = dati[dati['computation_type'] == computation_type]

    # Raggruppa i dati per numero di thread
    thread_values = sorted(set(valori['threads_used']))

    # Usa una palette di colori più grande
    colori = cm.get_cmap("tab20", len(valori['file_name'].unique()))  # tab20 ha 20 colori distinti

    # Per ogni file/matrice, tracciamo la curva
    for idx, file_name in enumerate(valori['file_name'].unique()):
        matrice_data = valori[valori['file_name'] == file_name]
        
        # Se il kernel è tra quelli che devono moltiplicare i thread per 32
        if computation_type in kernel_moltiplica_32:
            matrice_data['threads_used'] = matrice_data['threads_used'] * 32

        # Ordina i dati per numero di thread
        matrice_data = matrice_data.sort_values('threads_used')

        # Tracciamo la curva (linea continua) con un colore distintivo
        matrice_name = file_name.replace('.csv', '')  # Rimuove ".csv" dal nome della matrice
        plt.plot(matrice_data['threads_used'], matrice_data['GFLOPS'], marker='o', label=matrice_name, color=colori(idx))

    # Impostazioni del grafico
    plt.title(f"Prestazioni {computation_type} vs. Numero di Thread")
    plt.xlabel("Numero di Thread")
    plt.ylabel("Prestazioni (GFLOPS)")
    plt.grid(True)
    plt.tight_layout()

    # Posizioniamo la legenda fuori dal grafico a destra
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Matrice", fontsize=9)

    # Salva il grafico
    nome_file = f"{computation_type}_individual_performance.png"
    plot_path = os.path.join(output_dir, nome_file)
    plt.savefig(plot_path, bbox_inches='tight')  # bbox_inches='tight' per evitare il taglio della legenda
    plt.close()

    print(f"Grafico salvato: {plot_path}")

# Genera il grafico per ogni tipo di computazione
for tipo in tipi_computazione:
    genera_grafico_kernel(tipo)
