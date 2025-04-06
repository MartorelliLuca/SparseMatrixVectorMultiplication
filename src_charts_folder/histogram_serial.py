import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV
df = pd.read_csv('filtered_results.csv')

# Calcola il GFLOPS medio
gflops_medio = df['GFLOPS'].mean()
print(f'GFLOPS medio: {gflops_medio:.2f}')

# Crea l'istogramma
plt.figure(figsize=(10, 6))
plt.bar(df['name'], df['GFLOPS'], color='darkblue')

# Aggiungi una linea orizzontale per il GFLOPS medio
plt.axhline(y=gflops_medio, color='red', linestyle='--', label=f'Media GFLOPS: {gflops_medio:.2f}')

# Aggiungi etichette e titolo
plt.xlabel('Nome Matrice')
plt.ylabel('GFLOPS')
plt.title('GFLOPS per ciascuna matrice')
plt.legend()

# Ruota le etichette sull'asse x per migliorare la leggibilità
plt.xticks(rotation=45, ha='right')

# Ottimizza layout
plt.tight_layout()

# Salva il grafico
plt.savefig('grafico_GFLOPS.png', format='png')
print("Grafico salvato come 'grafico_GFLOPS.png'")

# Mostra il grafico
#plt.show()
