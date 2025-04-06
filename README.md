# Sparse Matrix-Vector Multiplication

## Problema
Questo progetto riguarda la realizzazione di un nucleo di calcolo per il prodotto tra una matrice sparsa e un vettore, ovvero:
\( y \leftarrow Ax \)

dove \( A \) è una matrice sparsa memorizzata nei formati:
1. **CSR (Compressed Sparse Row)**
2. **HLL (Hybrid Linear List)**

Il nucleo di calcolo sarà parallelizzato per sfruttare le risorse disponibili, utilizzando **OpenMP** e **CUDA**. Il codice sarà sviluppato in **C** e verrà testato confrontandolo con un'implementazione seriale di riferimento (suggerita in formato CSR). Saranno necessarie funzioni ausiliarie per:
- Il preprocessamento dei dati della matrice.
- La memorizzazione nei formati richiesti.
- La misurazione delle prestazioni su un server disponibile nel dipartimento.

---

## Matrici di Test
Le matrici utilizzate per il collaudo sono disponibili nella **Suite Sparse Matrix Collection**: [Sparse Matrix Collection](https://sparse.tamu.edu/). Si consiglia di scaricare i dati nel formato **MatrixMarket**.

Alcuni dettagli importanti:
- Le matrici simmetriche devono essere ricostruite completamente in memoria.
- Alcune matrici sono salvate come "pattern", dove tutti i coefficienti non nulli sono 1.0.
- I test includeranno almeno le seguenti matrici:
  - `cage4`, `Cube Coup dt0`, `FEM 3D thermal1`
  - `mhda416`, `ML Laplace`, `mcfe`, `bcsstk17`, `olm1000`
  - `mac econ fwd500`, `adder dcop 32`, `mhd4800a`, `nlpkkt80`
  - `west2021`, `cop20k A`, `webbase-1M`, `cavity10`
  - `raefsky2`, `dc1`, `rdist2`, `af23560`, `amazon0302`
  - `cant`, `lung2`, `af 1 k101`, `olafu`, `PR02R`, `roadNet-PA`

Ulteriori test possono essere eseguiti con altre matrici.

---

## Misurazione delle Prestazioni
Le misure di prestazione saranno ottenute ripetendo il calcolo del prodotto matrice-vettore più volte per ogni matrice, calcolando il tempo medio per esecuzione.

La misura delle prestazioni in **MFLOPS** o **GFLOPS** sarà:
\[
FLOPS = \frac{2 \times NZ}{T}
\]
Dove:
- \( NZ \) è il numero di elementi non nulli nella matrice.
- \( T \) è il tempo medio di esecuzione.

**Nota:**
- Il tempo di preprocessamento, I/O e trasferimento dati su GPU **non** sarà incluso nella misura principale (ma può essere discusso separatamente).
- Per la versione OpenMP, il codice verrà testato con un numero variabile di **thread**, da 1 fino al massimo numero di core disponibili.

---

## Formati di Memorizzazione
### 1. CSR (Compressed Sparse Row)
Memorizza una matrice \(M \times N\) con \( NZ \) elementi non nulli utilizzando:
- **M**: Numero di righe
- **N**: Numero di colonne
- **IRP(1:M+1)**: Puntatori all'inizio di ciascuna riga
- **JA(1:NZ)**: Indici di colonna
- **AS(1:NZ)**: Valori dei coefficienti

Esempio per la matrice:
```
11 12  0  0
 0 22 23  0
 0  0 33  0
 0  0 43 44
```
In formato CSR:
```
M = 4
N = 4
IRP = [1, 3, 5, 6, 8]
JA =  [1, 2, 2, 3, 3, 3, 4]
AS =  [11, 12, 22, 23, 33, 43, 44]
```

### 2. HLL (Hybrid Linear List)
- Definisce un parametro **HackSize** (es. 32).
- Divide la matrice in blocchi di **HackSize righe**.
- Ogni blocco è memorizzato in formato **ELLPACK**.

### 3. ELLPACK
Memorizza una matrice  \(M \times N\) con un massimo di \( MAXNZ \) non-zeri per riga usando:
- **M**: Numero di righe
- **N**: Numero di colonne
- **MAXNZ**: Massimo numero di non-zeri per riga
- **JA(1:M,1:MAXNZ)**: Indici di colonna (array 2D)
- **AS(1:M,1:MAXNZ)**: Coefficienti (array 2D)

Esempio per la matrice precedente con **MAXNZ = 2**:
```
JA =
[1  2]
[2  3]
[3  3]
[3  4]

AS =
[11 12]
[22 23]
[33  0]
[43 44]
```
Se una riga ha meno di **MAXNZ** valori, gli elementi rimanenti vengono riempiti con **zeri**.

---

## Struttura del Progetto
- `src/`: Codice sorgente C,
- `CUDA_include/`: Codice dei Kernel Cuda e rispettivi header,
- `CUDA_include/`: File '.cu' per l'invocazione dei kernel CUDA,
- `data/`: Direcotry contente i dati raccolti dalle varie iterazioni,
- `src_charts_folder`: Sorgenti python utilizzati per generare i grafici,
- `charts/`: Direcotry contente i grafici generati.

Il file 'clear.sh' rimuove la cartella 'build' generata in fase di compilazione.
Il file 'run.sh' viene utilizzato per la compilazione e la successiva esecuzione del progetto.
---

## Istruzioni per l'Esecuzione
### Compilazione ed Esecuzione
Per compilare ed eseguire il progetto, basta eseguire lo script `run.sh`.  
È necessario aver installato una versione di **CMake ≥ 3.10** e i driver **NVIDIA**.


---

## Autori
- **Luca Martorelli** - [GitHub Profile](https://github.com/MartorelliLuca)
- **Alessandro Cortese** - [GitHub Profile](https://github.com/alessandro-cortese)

---

