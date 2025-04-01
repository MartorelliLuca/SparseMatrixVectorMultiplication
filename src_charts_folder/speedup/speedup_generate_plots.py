import os
import pandas as pd
import matplotlib.pyplot as plt

# Percorso del file CSV generato
speedup_csv_path = os.path.join(os.path.dirname(__file__), "../../data/speedup_single_csv/speedup_results.csv")

# Percorso della directory di output per i grafici
dir_plots = os.path.join(os.path.dirname(__file__), "../../charts/speedup/speedup_single_plot")
os.makedirs(dir_plots, exist_ok=True)

# Caricare il CSV
df = pd.read_csv(speedup_csv_path)

# Ottenere le matrici uniche
matrices = df["name_matrix"].unique()

# Generare e salvare i plot per ogni matrice
for matrix in matrices:
    matrix_df = df[df["name_matrix"] == matrix]
    
    plt.figure(figsize=(10, 6))
    for comp_type in matrix_df["computation_type"].unique():
        subset = matrix_df[matrix_df["computation_type"] == comp_type]
        plt.plot(subset["num_threads"], subset["speedup"], marker='o', linestyle='-', label=comp_type)
    
    plt.xlabel("Numero di Thread")
    plt.ylabel("Speedup")
    plt.title(f"Speedup per {matrix}")
    plt.legend()
    plt.grid(True)
    
    # Salvare il grafico
    plot_path = os.path.join(dir_plots, f"{matrix}_speedup.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Grafico salvato: {plot_path}")

# Generare e salvare i plot delle medie generali
mean_plots = {
    "parallel_open_mp": ["parallel_open_mp_csr", "parallel_open_mp_hll"],
    "cuda_csr": ["cuda_csr_kernel_1", "cuda_csr_kernel_2", "cuda_csr_kernel_3", "cuda_csr_kernel_4"],
    "cuda_hll": ["cuda_hll_kernel_1", "cuda_hll_kernel_2", "cuda_hll_kernel_3", "cuda_hll_kernel_4"]
}

dir_plots2 = os.path.join(os.path.dirname(__file__), "../../charts/speedup/general_charts")
os.makedirs(dir_plots2, exist_ok=True)

for plot_name, computation_types in mean_plots.items():
    plt.figure(figsize=(10, 6))
    mean_df = df[df["computation_type"].isin(computation_types)]
    mean_speedup = mean_df.groupby(["num_threads", "computation_type"])["speedup"].mean().unstack()
    
    for comp_type in computation_types:
        if comp_type in mean_speedup:
            plt.plot(mean_speedup.index, mean_speedup[comp_type], marker='o', linestyle='-', label=comp_type)
    
    plt.xlabel("Numero di Thread")
    plt.ylabel("Speedup Medio")
    plt.title(f"Speedup Medio per {plot_name}")
    plt.legend()
    plt.grid(True)
    
    # Salvare il grafico
    plot_path = os.path.join(dir_plots2, f"{plot_name}_mean_speedup.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Grafico salvato: {plot_path}")

# ---------------------------- NUOVA FUNZIONALITÀ ----------------------------

# Directory per i nuovi plot basati su non_zero_values
dir_plots3 = os.path.join(os.path.dirname(__file__), "../../charts/speedup/non_zero_charts")
os.makedirs(dir_plots3, exist_ok=True)

# Definizione degli intervalli di non_zero_values
soglie = [0, 10000, 100000, 500000, 1000000, 2500000, 10000000, float('inf')]
etichette_soglie = [
    "non_zero_values <10000",
    "non_zero_values 10000-100000",
    "non_zero_values 100000-500000",
    "non_zero_values 500000-1000000",
    "non_zero_values 1000000-2500000",
    "non_zero_values 2500000-10000000",
    "non_zero_values >=10000000"
]

# Lista dei formati per cui generare i plot
formati = [
    "parallel_open_mp_csr", "parallel_open_mp_hll",
    "cuda_csr_kernel_1", "cuda_csr_kernel_2", "cuda_csr_kernel_3", "cuda_csr_kernel_4",
    "cuda_hll_kernel_1", "cuda_hll_kernel_2", "cuda_hll_kernel_3", "cuda_hll_kernel_4"
]

# Aggiungere una nuova colonna per categorizzare le matrici nei vari intervalli
df["non_zero_category"] = pd.cut(df["non_zero_values"], bins=soglie, labels=etichette_soglie, right=False)

# Generazione dei plot
for formato in formati:
    plt.figure(figsize=(12, 7))
    
    df_formato = df[df["computation_type"] == formato]
    
    # Calcolare la media dello speedup per ciascun intervallo e numero di thread
    media_speedup = df_formato.groupby(["num_threads", "non_zero_category"])["speedup"].mean().unstack()
    
    for categoria in etichette_soglie:
        if categoria in media_speedup:
            plt.plot(media_speedup.index, media_speedup[categoria], marker='o', linestyle='-', label=categoria)
    
    plt.xlabel("Numero di Thread")
    plt.ylabel("Speedup Medio")
    plt.title(f"Speedup Medio per {formato} in base a Non-Zero Values")
    plt.legend(title="Intervalli di Non-Zero Values", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Salvare il grafico
    plot_path = os.path.join(dir_plots3, f"{formato}_non_zero_speedup.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Grafico salvato: {plot_path}")