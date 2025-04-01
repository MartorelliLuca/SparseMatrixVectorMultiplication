import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../data")  # Modifica se necessario

# Directory di destinazione per i grafici
output_dir = os.path.join(os.path.dirname(__file__), "../charts/histogram/")  # Modifica il percorso come preferisci
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Crea la directory se non esiste

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))

def load_csv_files(file_list):
    """Carica specifici file CSV e restituisce una lista di dati con il nome del file."""
    data = []
    for file in file_list:
        df = pd.read_csv(file)
        data.append((os.path.basename(file), df))
    return data

def extract_data(csv_data):
    """Estrae i dati di interesse dai file CSV."""
    matrix_data = {}
    
    for filename, dataset in csv_data:
        for _, row in dataset.iterrows():
            name = row.get("computation_type")
            gigaFlops = row.get("GFLOPS")
            
            # Verifica se i dati necessari sono presenti
            if pd.isna(name) or pd.isna(gigaFlops):
                continue  # Salta la riga se i dati non sono validi
            
            if filename not in matrix_data:
                matrix_data[filename] = {}
            
            matrix_data[filename][name] = gigaFlops
    
    # Ordinare i file CSV (matrici) in ordine alfabetico
    sorted_matrices = sorted(matrix_data.keys())
    
    return matrix_data, sorted_matrices

def plot_gflops_from_csv(csv_data, sorted_matrices, save_path):
    """Crea grafici a barre separati per ogni tipo di computazione come cuda_csr e cuda_hll."""
    if not csv_data:
        print("Nessun dato valido trovato per generare il grafico.")
        return

    # Gruppo di computation_type per cuda_csr e cuda_hll
    cuda_csr_computations = ["cuda_csr_kernel_1", "cuda_csr_kernel_2", "cuda_csr_kernel_3", "cuda_csr_kernel_4"]
    cuda_hll_computations = ["cuda_hll_kernel_1", "cuda_hll_kernel_2", "cuda_hll_kernel_3", "cuda_hll_kernel_4"]
    openmp_computations = ["parallel_open_mp_csr", "parallel_open_mp_hll"]

    # Colori specifici per ogni tipo di computation
    computation_colors = {
        "cuda_csr_kernel_1": "darkblue",
        "cuda_csr_kernel_2": "steelblue",
        "cuda_csr_kernel_3": "skyblue",
        "cuda_csr_kernel_4": "powderblue",
        "cuda_hll_kernel_1": "darkblue",
        "cuda_hll_kernel_2": "steelblue",
        "cuda_hll_kernel_3": "skyblue",
        "cuda_hll_kernel_4": "powderblue",
        "parallel_open_mp_csr": "darkblue",
        "parallel_open_mp_hll": "skyblue"
    }

    def create_graph(computation_types, title, file_suffix):
        """Crea il grafico per i dati di un gruppo specifico di computation_type (ad esempio, cuda_csr o cuda_hll)."""
        # Prepariamo i valori per il grafico: ogni matrice avrà un valore per ogni computation_type
        values = np.array([[csv_data[matrix].get(computation, 0) for computation in computation_types] for matrix in sorted_matrices])

        # Crea il grafico con una dimensione maggiore
        plt.figure(figsize=(20, 10))  # Aumenta la larghezza e altezza del grafico per avere più spazio

        bar_width = 0.2  # Larghezza della barra
        space_between_matrices = 0.5  # Distanziamento tra gruppi di matrici
        indices = np.arange(len(sorted_matrices))  # Indici per la posizione delle barre

        # Aggiungi le barre per ciascun tipo di computazione con i colori specificati
        for i, computation_type in enumerate(computation_types):
            bar_values = values[:, i]  # I GFLOPS per il computation_type corrente
            color = computation_colors[computation_type]  # Colore specifico per il computation_type
            # Le barre sono attaccate tra loro, con uno spazio tra matrici differenti
            plt.bar(indices + i * (bar_width), bar_values, width=bar_width, label=computation_type, color=color)

        # Configura le etichette sull'asse x
        xticks_positions = indices + (bar_width * len(computation_types)) / 2
        plt.xticks(xticks_positions, sorted_matrices, rotation=45, fontsize=12, ha='right')  # Aggiusta l'orientamento delle etichette

        plt.xlabel("Matrice", fontsize=14)
        plt.ylabel("Gigaflops", fontsize=14)
        plt.title(title, fontsize=16)

        # Legenda
        plt.legend(title="Computation Type", fontsize=12)

        # Aggiungi una griglia per migliorare la leggibilità
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Aggiusta il layout per evitare sovrapposizioni
        plt.tight_layout()

        # Salvataggio del grafico nella cartella specificata
        output_file = os.path.join(save_path, f"{file_suffix}_histogram.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Grafico {file_suffix} salvato in: {output_file}")

    # Crea i grafici separati per cuda_csr, cuda_hll, e openmp
    create_graph(cuda_csr_computations, "Prestazioni CUDA CSR", "cuda_csr")
    create_graph(cuda_hll_computations, "Prestazioni CUDA HLL", "cuda_hll")
    create_graph(openmp_computations, "Prestazioni OpenMP", "openmp")

# Esecuzione dello script con i file trovati
csv_data = load_csv_files(csv_files)
matrix_data, sorted_matrices = extract_data(csv_data)
plot_gflops_from_csv(matrix_data, sorted_matrices, output_dir)