import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../data2/data")  # Modifica se necessario

# Directory di destinazione per i grafici
output_dir_1 = os.path.join(os.path.dirname(__file__), "../charts/histogram/mean_histogram/")  # Flusso 1
output_dir_2 = os.path.join(os.path.dirname(__file__), "../charts/histogram/single_histogram/")  # Flusso 2
output_dir_3 = os.path.join(os.path.dirname(__file__), "../charts/histogram/average_gflops/")  # Nuovo grafico per la media separata
for dir_path in [output_dir_1, output_dir_2, output_dir_3]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Crea la directory se non esiste

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
            threads = row.get("threads_used")  # Assumiamo che il numero di thread sia una colonna nel CSV
            
            # Verifica se i dati necessari sono presenti
            if pd.isna(name) or pd.isna(gigaFlops) or pd.isna(threads):
                continue  # Salta la riga se i dati non sono validi
            
            if filename not in matrix_data:
                matrix_data[filename] = {}
            if name not in matrix_data[filename]:
                matrix_data[filename][name] = {}

            matrix_data[filename][name][threads] = gigaFlops  # Aggiungi i GFLOPS per quel numero di thread
    
    # Ordinare i file CSV (matrici) in ordine alfabetico
    sorted_matrices = sorted(matrix_data.keys())
    
    return matrix_data, sorted_matrices

def plot_gflops_by_threads_single(csv_data, sorted_matrices, save_path):
    """Crea un istogramma considerando solo cuda_csr_kernel_4 e cuda_hll_kernel_4."""
    if not csv_data:
        print("Nessun dato valido trovato per generare il grafico.")
        return

    # I tipi di computazione da analizzare
    computation_types = ["cuda_csr_kernel_4", "cuda_hll_kernel_4"]

    # Colori specifici per ogni tipo di computation
    computation_colors = {
        "cuda_csr_kernel_4": "darkblue",
        "cuda_hll_kernel_4": "skyblue"
    }

    # Funzione per creare un istogramma per ogni gruppo di computation_type
    def create_histogram(matrix, computation_types, title, file_suffix):
        """Crea un istogramma per un gruppo di computation_type per una matrice."""
        matrix_data = csv_data[matrix]
        thread_counts = sorted({thread for comp in computation_types for thread in matrix_data.get(comp, {})})
        
        if not thread_counts:
            print(f"Nessun dato valido per {matrix} ({file_suffix})")
            return

        bar_width = 0.15  # Larghezza delle barre
        indices = np.arange(len(thread_counts))  # Indici per i gruppi di thread

        plt.figure(figsize=(20, 10))  # Imposta la dimensione del grafico

        # Disegna le barre per ogni computation_type
        for i, computation_type in enumerate(computation_types):
            computation_data = matrix_data.get(computation_type, {})
            gflops_values = [computation_data.get(thread, 0) for thread in thread_counts]
            plt.bar(indices + i * bar_width, gflops_values, width=bar_width, label=computation_type, color=computation_colors.get(computation_type, "gray"))

        # Configura le etichette sull'asse x
        plt.xticks(indices + (bar_width * len(computation_types)) / 2, thread_counts, fontsize=12)
        plt.xlabel("Numero di Thread", fontsize=14)
        plt.ylabel("Gigaflops", fontsize=14)
        plt.title(f"{title} - Matrice {matrix}", fontsize=16)

        # Legenda
        plt.legend(title="Computation Type", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Salvataggio del grafico per quella matrice
        output_file = os.path.join(save_path, f"{matrix}_{file_suffix}_histogram.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Istogramma per {matrix} ({file_suffix}) salvato in: {output_file}")

    # Crea gli istogrammi per ogni matrice e per ogni tipo di computazione
    for matrix in sorted_matrices:
        create_histogram(matrix, computation_types, "Prestazioni GPU CUDA e HLL", "cuda_csr_cuda_hll")

def plot_average_gflops(csv_data, sorted_matrices, save_path):
    """Crea un grafico della media separata dei GFLOPS per cuda_csr_kernel_4 e cuda_hll_kernel_4."""
    avg_gflops_csr = []
    avg_gflops_hll = []

    # Calcolare la media dei GFLOPS separatamente per i due kernel
    for matrix in sorted_matrices:
        matrix_data = csv_data[matrix]

        # Media per cuda_csr_kernel_4
        csr_gflops = []
        for gflops in matrix_data.get("cuda_csr_kernel_4", {}).values():
            csr_gflops.append(gflops)
        avg_csr = np.mean(csr_gflops) if csr_gflops else 0

        # Media per cuda_hll_kernel_4
        hll_gflops = []
        for gflops in matrix_data.get("cuda_hll_kernel_4", {}).values():
            hll_gflops.append(gflops)
        avg_hll = np.mean(hll_gflops) if hll_gflops else 0

        avg_gflops_csr.append(avg_csr)
        avg_gflops_hll.append(avg_hll)

    # Creare il grafico della media separata dei GFLOPS per ciascun kernel
    x = np.arange(len(sorted_matrices))  # Posizioni delle matrici
    width = 0.35  # Larghezza delle barre

    plt.figure(figsize=(20, 10))
    plt.bar(x - width / 2, avg_gflops_csr, width, label="cuda_csr_kernel_4", color='darkblue')
    plt.bar(x + width / 2, avg_gflops_hll, width, label="cuda_hll_kernel_4", color='skyblue')

    # Etichette e titolo
    plt.xlabel("Matrice", fontsize=14)
    plt.ylabel("Media dei GFLOPS", fontsize=14)
    plt.title("Media dei GFLOPS per ciascun kernel (cuda_csr_kernel_4 vs cuda_hll_kernel_4)", fontsize=16)
    plt.xticks(x, sorted_matrices, rotation=45, ha='right', fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Salvataggio del grafico
    output_file = os.path.join(save_path, "average_gflops_comparison.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Grafico della media dei GFLOPS separata per ciascun kernel salvato in: {output_file}")

# Esecuzione del flusso
csv_data = load_csv_files(csv_files)
matrix_data, sorted_matrices = extract_data(csv_data)

# Flusso: Calcola i GFLOPS per ogni numero di thread e genera gli istogrammi solo per cuda_csr_kernel_4 e cuda_hll_kernel_4
plot_gflops_by_threads_single(matrix_data, sorted_matrices, output_dir_2)

# Nuovo grafico per la media separata dei GFLOPS per ciascun kernel
plot_average_gflops(matrix_data, sorted_matrices, output_dir_3)
