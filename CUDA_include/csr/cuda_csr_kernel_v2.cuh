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

__global__ void csr_matvec_warp_shmem(CSR_matrix d_A, double *d_x, double *d_y)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // ID del thread globale
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if (warp_id >= d_A.M)
        return;

    __shared__ double shared_vals[WARP_SIZE]; // Setto la shared memory per riduzione
    double sum = 0.0;
    int row_start = d_A.IRP[warp_id];
    int row_end = d_A.IRP[warp_id + 1];
    for (int i = row_start + lane; i < row_end; i += WARP_SIZE)
    {
        sum += d_A.AS[i] * d_x[d_A.JA[i]];
    }

    shared_vals[lane] = sum; // Salvo la somma parziale nella memoria condivisa
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) // Riduzione in memoria condivisa
    {
        if (lane < offset)
        {
            shared_vals[lane] += shared_vals[lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0) // Il primo thread del warp scrive il risultato nella memoria globale
    {
        d_y[warp_id] = shared_vals[0];
    }
}