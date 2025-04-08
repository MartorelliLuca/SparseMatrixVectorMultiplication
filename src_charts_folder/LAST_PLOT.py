import os
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Caricamento dei dati ===

# Percorso della cartella "data"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")

# Leggo tutti i CSV nella cartella
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

# Controllo se ci sono file
if not csv_files:
    raise FileNotFoundError(f"Nessun file CSV trovato nella cartella: {data_dir}")

# Concateno tutti i dati in un unico DataFrame
dati_list = []
for file in csv_files:
    df = pd.read_csv(file)
    # Verifica che ci siano le colonne richieste
    if {"threads_used", "computation_type", "GFLOPS"}.issubset(df.columns):
        df["threads_used"] = df["threads_used"].astype(int)
        dati_list.append(df)
    else:
        print(f"Attenzione: file ignorato perché mancano colonne richieste -> {file}")

# Controllo se ci sono dati validi
if not dati_list:
    raise ValueError("Non ci sono dati validi nei file CSV.")

# Creo il DataFrame finale
dati = pd.concat(dati_list, ignore_index=True)

# === 2. Configurazioni richieste ===

configurazioni = {
    "cuda_csr_kernel_1 (128 thread)": ("cuda_csr_kernel_1", 3),
    "cuda_csr_kernel_2 (128 thread)": ("cuda_csr_kernel_2", 3),
    "cuda_csr_kernel_3 (128 thread)": ("cuda_csr_kernel_3", 4),
    "cuda_csr_kernel_4 (128 thread)": ("cuda_csr_kernel_4", 4)
}

# === 3. Calcolo GFLOPS medi ===

media_gflops = {}
for label_legenda, (kernel, thread_count) in configurazioni.items():
    valori = dati[(dati['computation_type'] == kernel) & (dati['threads_used'] == thread_count)]
    if not valori.empty:
        media = valori['GFLOPS'].mean()
        media_gflops[label_legenda] = media
    else:
        print(f"⚠️ Nessun dato trovato per {label_legenda}. Verrà impostato a 0.")
        media_gflops[label_legenda] = 0

# === 4. Plot ===

plt.figure(figsize=(8, 6))
plt.bar(
    media_gflops.keys(),
    media_gflops.values(),
    color='skyblue',
    edgecolor='black',      # Bordo nero
    linewidth=1.0           # Spessore bordo sottile
)

plt.title("Confronto GFLOPS medi tra diverse configurazioni kernel")
plt.ylabel("GFLOPS medi")
plt.xticks(rotation=20)

# Aggiungi i valori sopra le barre
for i, (label, value) in enumerate(media_gflops.items()):
    plt.text(i, value + 0.1, f"{value:.2f}", ha='center', va='bottom')

plt.tight_layout()

# Salva il grafico
output_dir = os.path.join(script_dir, "../charts/performance/mean_performance_plot")
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, "confronto_gflops_configurazioni.png")
plt.savefig(plot_path)
plt.close()

print(f"✅ Grafico salvato correttamente in: {plot_path}")
