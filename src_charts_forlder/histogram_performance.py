import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "data")  # Modifica se necessario

# Directory di destinazione per i grafici
output_dir = os.path.join(os.path.dirname(__file__), "../charts/histogram/histograms111")  # Modifica il percorso come preferisci
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
    nonzeros = {}
    
    for filename, dataset in csv_data:
        for _, row in dataset.iterrows():
            name = row["computation_type"]
            megaFlops = row["MFLOPS"]
            nonzero_count = row["non_zero_values"]
            
            if name not in matrix_data:
                matrix_data[name] = {}
                nonzeros[name] = nonzero_count
            
            matrix_data[name][filename] = megaFlops
    
    # Ordina le matrici in base al numero di nonzeri
    sorted_matrices = sorted(matrix_data.keys(), key=lambda x: nonzeros[x])
    return matrix_data, sorted_matrices

def plot_histogram(matrix_data, sorted_matrices, save_path):
    """Crea un grafico a istogrammi comparando i megaflops, ordinando per numero di nonzeri e salva l'immagine."""
    if not matrix_data:
        print("Nessun dato valido trovato per generare il grafico.")
        return

    file_names = sorted({file for values in matrix_data.values() for file in values})
    
    if not file_names:
        print("Nessun file con dati validi trovato.")
        return

    values = [[matrix_data[matrix].get(file, 0) for file in file_names] for matrix in sorted_matrices]
    
    plt.figure(figsize=(14, 7))
    bar_width = 0.2
    indices = np.arange(len(sorted_matrices))

    for i, file_name in enumerate(file_names):
        bar_values = [v[i] for v in values if len(v) > i]  # Evita errori di indice
        if any(bar_values):  # Aggiunge la barra solo se ha dati
            plt.bar(indices + i * bar_width, bar_values, width=bar_width, label=file_name)

    plt.xticks(indices + (bar_width * (len(file_names) / 2)), sorted_matrices, rotation=90, fontsize=12)
    plt.xlabel("Tipo di Computazione", fontsize=14)
    plt.ylabel("Megaflops", fontsize=14)
    plt.title("Confronto delle Prestazioni del Prodotto Matrice-Vettore", fontsize=16)

    if plt.gca().get_legend_handles_labels()[0]:  # Controlla se ci sono etichette valide
        plt.legend(fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvataggio del grafico nella cartella specificata
    output_file = os.path.join(save_path, "confronto_prestazioni.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Grafico salvato in: {output_file}")

# Esecuzione dello script con i file trovati
csv_data = load_csv_files(csv_files)
matrix_data, sorted_matrices = extract_data(csv_data)
plot_histogram(matrix_data, sorted_matrices, output_dir)

