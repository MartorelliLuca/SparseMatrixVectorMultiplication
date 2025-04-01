import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#per calcolare l'efficienza fai il rapporto fra speedup e numero di thread


# Percorso della directory contenente i file CSV
dir_csv = os.path.join(os.path.dirname(__file__), "../../data")  

# Lista di tutti i file CSV nella directory
csv_files = glob.glob(os.path.join(dir_csv, "*.csv"))
if not csv_files:
    print("Errore: Nessun file CSV trovato nella directory!")
    exit()

# Soglie di nonzeri
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

# Dizionari per i dati
dati_serial = {}
dati_parallel = {}
num_nonzeri = {}

# Caricamento dati dai CSV
for file_path in csv_files:
    df = pd.read_csv(file_path)

    # Debug: Controlliamo che le colonne esistano
    expected_columns = {"computation_type", "non_zero_values", "threads_used", "time_used"}
    if not expected_columns.issubset(df.columns):
        print(f"Errore: Colonne mancanti nel file {file_path}. Trovate: {df.columns}")
        continue

    for _, row in df.iterrows():
        name_matrix = row["computation_type"]
        nonzeros = row["non_zero_values"]
        num_threads = row["threads_used"]
        time_used = row["time_used"]

        num_nonzeri[name_matrix] = nonzeros  # Salviamo il numero di nonzeri

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

# Debug: Verifica che i dati seriali e paralleli siano stati caricati
print(f"Dati seriali caricati: {dati_serial}")
print(f"Dati paralleli caricati: {dati_parallel}")

# Calcolo dell'efficienza
matrici_efficiency = {}
for matrix_name in dati_serial:
    if matrix_name in dati_parallel:
        matrici_efficiency[matrix_name] = {}
        for num_thread in dati_parallel[matrix_name]:
            if num_thread in dati_serial[matrix_name]:
                serial_time = dati_serial[matrix_name][num_thread]
                parallel_time = dati_parallel[matrix_name][num_thread]

                if parallel_time <= 0:
                    print(f"Attenzione! Tempo di esecuzione parallelo zero o negativo per {matrix_name} con {num_thread} thread.")
                    continue

                efficiency = serial_time / (parallel_time * num_thread)
                matrici_efficiency[matrix_name][num_thread] = efficiency
            else:
                print(f"Errore: Numero di thread {num_thread} non trovato in dati seriali per {matrix_name}.")
    else:
        print(f"Errore: Nessun dato parallelo trovato per {matrix_name}.")

# Debug: Verifica che ci siano efficienze calcolate
print(f"Efficienze calcolate: {matrici_efficiency}")

# Raggruppamento per soglia di nonzeri
gruppi_efficiency = {s: {} for s in soglie_nonzeri}

for matrix_name, efficiencies in matrici_efficiency.items():
    nonzeros = num_nonzeri.get(matrix_name, None)
    if nonzeros is None:
        print(f"Attenzione! Matrice {matrix_name} non ha un numero di nonzeri definito.")
        continue

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

# Calcolare efficienza media
efficiency_medio = {}
for soglia, efficiency_threads in gruppi_efficiency.items():
    efficiency_medio[soglia] = {
        num_thread: np.mean(valori_efficiency)
        for num_thread, valori_efficiency in efficiency_threads.items() if valori_efficiency
    }

# Debug: Verifica se ci sono dati validi per il plot
print(f"Efficienza media per soglia: {efficiency_medio}")
if not any(efficiency_medio.values()):
    print("Errore: Nessun dato di efficienza calcolato, impossibile plottare!")
    exit()

# Creazione del grafico
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