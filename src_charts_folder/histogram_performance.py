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
for dir_path in [output_dir_1, output_dir_2]:
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

def plot_gflops_by_threads_avg(csv_data, sorted_matrices, save_path):
    """Calcola la media dei GFLOPS per ogni numero di thread e crea i grafici separati per computation_type."""
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

    def create_avg_graph(computation_types, title, file_suffix):
        """Crea il grafico per la media dei GFLOPS per numero di thread."""
        # Per ogni matrice, calcoliamo la media dei GFLOPS per ogni numero di thread
        values = []
        threads_list = []

        for matrix in sorted_matrices:
            matrix_values = []
            for computation in computation_types:
                computation_data = csv_data[matrix].get(computation, {})
                if computation_data:
                    # Calcoliamo la media dei GFLOPS per ogni numero di thread
                    avg_gflops = np.mean(list(computation_data.values()))
                    matrix_values.append(avg_gflops)
                    threads_list = sorted(computation_data.keys())  # Teniamo traccia dei thread usati
                else:
                    matrix_values.append(0)  # Se non ci sono dati, aggiungiamo 0

            values.append(matrix_values)

        # Crea il grafico con una dimensione maggiore
        plt.figure(figsize=(20, 10))  # Aumenta la larghezza e altezza del grafico per avere più spazio

        bar_width = 0.2  # Larghezza della barra
        indices = np.arange(len(sorted_matrices))  # Indici per la posizione delle barre

        # Aggiungi le barre per ciascun tipo di computazione con i colori specificati
        for i, computation_type in enumerate(computation_types):
            bar_values = [values[j][i] for j in range(len(values))]  # I GFLOPS medi per il computation_type
            color = computation_colors[computation_type]  # Colore specifico per il computation_type
            plt.bar(indices + i * (bar_width), bar_values, width=bar_width, label=computation_type, color=color)

        # Configura le etichette sull'asse x
        xticks_positions = indices + (bar_width * len(computation_types)) / 2
        plt.xticks(xticks_positions, sorted_matrices, rotation=45, fontsize=12, ha='right')  # Aggiusta l'orientamento delle etichette

        plt.xlabel("Matrice", fontsize=14)
        plt.ylabel("Gigaflops Medi", fontsize=14)
        plt.title(title, fontsize=16)

        # Legenda
        plt.legend(title="Computation Type", fontsize=12)

        # Aggiungi una griglia per migliorare la leggibilità
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Aggiusta il layout per evitare sovrapposizioni
        plt.tight_layout()

        # Salvataggio del grafico nella cartella specificata
        output_file = os.path.join(save_path, f"{file_suffix}_avg_histogram.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Grafico {file_suffix} salvato in: {output_file}")

    # Crea i grafici separati per cuda_csr, cuda_hll, e openmp
    create_avg_graph(cuda_csr_computations, "Prestazioni Medie CUDA CSR", "cuda_csr")
    create_avg_graph(cuda_hll_computations, "Prestazioni Medie CUDA HLL", "cuda_hll")
    create_avg_graph(openmp_computations, "Prestazioni Medie OpenMP", "openmp")


def plot_gflops_by_threads(csv_data, sorted_matrices, save_path):
    """Crea 3 grafici separati per ogni matrice in base a computation_type."""
    if not csv_data:
        print("Nessun dato valido trovato per generare il grafico.")
        return

    # Gruppo di computation_type
    parallel_open_mp_computations = ["parallel_open_mp_csr", "parallel_open_mp_hll"]
    cuda_csr_computations = ["cuda_csr_kernel_1", "cuda_csr_kernel_2", "cuda_csr_kernel_3", "cuda_csr_kernel_4"]
    cuda_hll_computations = ["cuda_hll_kernel_1", "cuda_hll_kernel_2", "cuda_hll_kernel_3", "cuda_hll_kernel_4"]

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

    # Crea gli istogrammi per ogni matrice e per ogni gruppo di computation_type
    for matrix in sorted_matrices:
        create_histogram(matrix, parallel_open_mp_computations, "Parallel OpenMP", "parallel_open_mp")
        create_histogram(matrix, cuda_csr_computations, "CUDA CSR", "cuda_csr")
        create_histogram(matrix, cuda_hll_computations, "CUDA HLL", "cuda_hll")

# Esecuzione dei due flussi
csv_data = load_csv_files(csv_files)
matrix_data, sorted_matrices = extract_data(csv_data)

# Flusso 1: Calcola la media dei GFLOPS e genera i grafici per computation_type
plot_gflops_by_threads_avg(matrix_data, sorted_matrices, output_dir_1)

# Flusso 2: Calcola i GFLOPS per ogni numero di thread e genera i grafici per ogni matrice
plot_gflops_by_threads(matrix_data, sorted_matrices, output_dir_2)
