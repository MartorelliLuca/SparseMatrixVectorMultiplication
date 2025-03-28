#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/csr_matrix.h"

/******************************************************************************************************************
 * Terzo Kernel CUDA, esso sfrutta:
 * Questo kernel suddivide il calcolo tra i warp dei thread e utilizza la memoria condivisa
 * per accumulare i risultati parziali, eseguendo poi una riduzione per ottenere il valore finale.

 * Struttura del codice:
 * - Ogni riga della matrice CSR è assegnata a un warp.
 * - I thread del warp caricano e moltiplicano i valori non nulli della riga con il vettore x.
 * - I risultati parziali vengono accumulati nella memoria condivisa. L'utilizzo della memoria condivisa ci permette di:
 *      1) Garantire una coalescenza degli accessi, migliorando l'efficienza della memoria.
 *      2) Avere una riduzione dei dati, in modo particolare della latenza nell'accesso a valori frequentemente utilizzati.
 * - La riduzione viene eseguita tramite un ciclo unroll manuale per migliorare le prestazioni.
 *
 ******************************************************************************************************************/

#define WARP_SIZE 32

__global__ void csr_matvec_shared_memory(CSR_matrix d_csr, int M, double *d_x, double *d_y)
{
    extern __shared__ double shared_data[];

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x; // Indice del thread nel warp
    if (row < M)
    {
        int start_row = d_csr.IRP[row];
        int end_row = d_csr.IRP[row + 1];

        double sum = 0.0;
        for (int i = start_row + lane; i < end_row; i += WARP_SIZE)
        {
            sum += d_csr.AS[i] * d_x[d_csr.JA[i]];
        }

        shared_data[lane + threadIdx.y * WARP_SIZE] = sum;
        __syncthreads();

        // Riduzione nella shared memory usando un ciclo unroll manuale
        if (lane < 16)
            shared_data[lane + threadIdx.y * WARP_SIZE] += shared_data[lane + threadIdx.y * WARP_SIZE + 16];
        __syncthreads();
        if (lane < 8)
            shared_data[lane + threadIdx.y * WARP_SIZE] += shared_data[lane + threadIdx.y * WARP_SIZE + 8];
        __syncthreads();
        if (lane < 4)
            shared_data[lane + threadIdx.y * WARP_SIZE] += shared_data[lane + threadIdx.y * WARP_SIZE + 4];
        __syncthreads();
        if (lane < 2)
            shared_data[lane + threadIdx.y * WARP_SIZE] += shared_data[lane + threadIdx.y * WARP_SIZE + 2];
        __syncthreads();
        if (lane < 1)
            shared_data[lane + threadIdx.y * WARP_SIZE] += shared_data[lane + threadIdx.y * WARP_SIZE + 1];
        __syncthreads();

        if (lane == 0)
        {
            d_y[row] = shared_data[threadIdx.y * WARP_SIZE];
        }
    }
}
