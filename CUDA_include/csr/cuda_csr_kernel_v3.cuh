#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/csr_matrix.h"


//TODO ATTENZIONE QUESTO E' TUTTO BUGGATO, DA RIVEDERE BENE


/******************************************************************************************************************
 * Terzo Kernel CUDA, esso sfrutta:
 * - Registri: utilizzo della memoria più veloce disponibile per ogni thread.
 * - Shared Memory: riduzione degli accessi alla memoria globale per il vettore x.
 * - AtomicAdd: gestione sicura dell'aggiornamento del risultato evitando race condition.
 *
 * Struttura del kernel:
 * - Ogni warp (32 thread) calcola il prodotto riga-vettore per una riga della matrice CSR.
 * - I thread del warp caricano e processano i valori della riga in parallelo.
 * - Una riduzione parallela (warp shuffle) combina i risultati dei thread all'interno del warp.
 * - Il primo thread del warp aggiorna `d_y[row]` con `atomicAdd()`.
 *
 * Vantaggi:
 * - Riduzione degli accessi alla global memory → Migliore caching ed efficienza.
 * - Migliore parallelizzazione intra-warp → I thread collaborano meglio nel calcolo.
 * - Efficiente su matrici di grandi dimensioni → Evita latenze elevate sugli accessi alla memoria globale.
 *
 * Svantaggi:
 * - Necessità di sincronizzazione (`__syncthreads()`) → Aggiunge un piccolo overhead.
 ******************************************************************************************************************/

#define WARP_SIZE 32

__global__ void csr_matvec_atomic_warp_shmem(CSR_matrix d_csr, double *d_x, double *d_y)
{
    extern __shared__ double shared_x[]; // Shared memory per il vettore x
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;

    // Ogni thread carica un pezzo del vettore x in shared memory
    int col_idx = threadIdx.x + blockDim.x * warp_id;
    if (col_idx < d_csr.N)
    {
        shared_x[col_idx] = d_x[col_idx];
    }
    __syncthreads(); // Assicuriamoci che tutta la shared memory sia stata riempita

    if (row < d_csr.M)
    {
        int row_start = d_csr.IRP[row];
        int row_end = d_csr.IRP[row + 1];

        double sum = 0.0;
        for (int i = row_start + lane; i < row_end; i += WARP_SIZE)
        {
            sum += d_csr.AS[i] * shared_x[d_csr.JA[i]];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane == 0)
        {
            atomicAdd(&d_y[row], sum);
        }
    }
}
