#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/csr_matrix.h"

/******************************************************************************************************************
 * Calcola il prodotto matrice-vettore (SpMV) per una matrice sparsa in formato CSR,
 * utilizzando parallelismo a livello di warp e memoria condivisa per ottimizzare l'accesso ai dati.
 *
 * Ogni warp è responsabile di una riga della matrice. I thread all'interno del warp calcolano i prodotti parziali
 * dei valori non nulli della riga e li accumulano in memoria condivisa. Viene poi eseguita una riduzione parallela
 * per ottenere la somma finale della riga, che viene scritta nel vettore di output.
 *
 * Vantaggi:
 * - Migliore coalescenza nella lettura della memoria
 * - Riduzione delle latenze grazie alla shared memory e al parallelismo warp-level.
 * - Riduzione del numero di accessi alla memoria globale.
 *
 * Svantaggi:
 * - Aumento di latenza dovuta dalla scrittura sulla shared memory.
 ******************************************************************************************************************/
#define WARP_SIZE 32

__global__ void csr_matvec_warp_shmem(CSR_matrix d_mat, const double *x, double *y)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row < d_mat.M)
    {
        double sum = 0.0;
        int start_row = d_mat.IRP[row];
        int end_row = d_mat.IRP[row + 1];

        for (int j = start_row + lane; j < end_row; j += WARP_SIZE)
        {
            sum += d_mat.AS[j] * x[d_mat.JA[j]]; // Ogni thread processa elementi della riga
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_sync(0xFFFFFFFF, sum, lane - offset); // Riduzione warp-level usando __shfl_sync
        }

        if (lane == 0)
        { // Solo il primo thread del warp scrive il risultato finale
            y[row] = sum;
        }
    }
}