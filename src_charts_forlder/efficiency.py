import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Percorso corretto della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "data")  # Percorso relativo alla posizione dello script

# Lista di tutti i file CSV nella directory data
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))


# Definizione delle soglie di nonzeri
soglie_nonzeri = [10000, 100000, 500000, 1e6, 2.5e6, 1e7]
etichette_soglie = [
    "Nonzeri <10000",
    "Nonzeri 10000-100000",
    "Nonzeri 100000-500000",
    "Nonzeri 500000-1000000",
    "Nonzeri 1000000-2500000",
    "Nonzeri 2500000-10000000",
    "Nonzeri >=10000000"
]
colori = ['b', 'orange', 'g', 'r', 'purple', 'brown', 'pink']

# Dizionari per memorizzare i dati
dati_serial = {}
dati_parallel = {}
num_nonzeri = {}

# Caricare i dati dai file CSV
for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        name_matrix = row["computation_type"]
        nonzeros = row["non_zero_values"]
        num_threads = row["threads_used"]
        time_used = row["time_used"]
        
        num_nonzeri[name_matrix] = nonzeros

        # Dati seriali
        if "serial" in name_matrix:
            if name_matrix not in dati_serial:
                dati_serial[name_matrix] = {}
            dati_serial[name_matrix][num_threads] = time_used

        # Dati paralleli
        if "parallel" in name_matrix:
            if name_matrix not in dati_parallel:
                dati_parallel[name_matrix] = {}
            dati_parallel[name_matrix][num_threads] = time_used

# Calcolo dell'efficienza
matrici_efficiency = {}
for matrix_name in dati_serial:
    if matrix_name in dati_parallel:
        matrici_efficiency[matrix_name] = {}
        for num_thread in dati_parallel[matrix_name]:
            if num_thread in dati_serial[matrix_name]:
                serial_time = dati_serial[matrix_name][num_thread]
                parallel_time = dati_parallel[matrix_name][num_thread]
                if parallel_time > 0:
                    efficiency = serial_time / (parallel_time * num_thread)
                    matrici_efficiency[matrix_name][num_thread] = efficiency

# Raggruppare gli efficiency medi per soglie di nonzeri
gruppi_efficiency = {s: {} for s in soglie_nonzeri}

for matrix_name, efficiencies in matrici_efficiency.items():
    nonzeros = num_nonzeri[matrix_name]
    
    for i, soglia in enumerate(soglie_nonzeri):
        if nonzeros <= soglia:
            for num_thread, efficiency in efficiencies.items():
                if num_thread not in gruppi_efficiency[soglia]:
                    gruppi_efficiency[soglia][num_thread] = []
                gruppi_efficiency[soglia][num_thread].append(efficiency)
            break
    else:
        soglia = soglie_nonzeri[-1]
        for num_thread, efficiency in efficiencies.items():
            if num_thread not in gruppi_efficiency[soglia]:
                gruppi_efficiency[soglia][num_thread] = []
            gruppi_efficiency[soglia][num_thread].append(efficiency)

# Calcolare lo efficiency medio per ogni intervallo di nonzeri
efficiency_medio = {}
for soglia, efficiency_threads in gruppi_efficiency.items():
    efficiency_medio[soglia] = {
        num_thread: np.mean(valori_efficiency)
        for num_thread, valori_efficiency in efficiency_threads.items() if valori_efficiency
    }

# Creare il grafico
plt.figure(figsize=(12, 7))
for i, (soglia, efficiencies) in enumerate(efficiency_medio.items()):
    if efficiencies:
        thread_values = sorted(efficiencies.keys())
        efficiency_values = [efficiencies[t] for t in thread_values]
        plt.plot(thread_values, efficiency_values, linestyle='-', 
                 color=colori[i], label=etichette_soglie[i])

plt.title("Efficienza Media vs. Numero di Thread")
plt.xlabel("Numero di Thread")
plt.ylabel("Efficienza Media")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
