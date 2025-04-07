#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/csr_matrix.h"

/******************************************************************************************************************
 * Ogni warp è responsabile del calcolo di una singola riga della matrice.
 * - Utilizzo di shared memory (memoria condivisa) per garantire un accesso più efficiente ai dati:
 *       1) L'uso della shared memory permette accessi alla memoria coalescenti.
 * - Implementazione di una riduzione per sommare i risultati, utilizzando la funzione __shfl_sync:
 *       1) La funzione __shfl_sync permette una comunicazione diretta tra i thread di uno stesso warp.
 *       2) Utilizzare __shfl_sync riduce il bisogno di sincronizzare la memoria globale, migliorando l'efficienza.
 *       3) L'uso di __shfl_sync con variabili di tipo double può introdurre piccoli errori di accumulo nei calcoli a causa della precisione numerica limitata.
 ******************************************************************************************************************/
#define WARP_SIZE 32

__global__ void csr_matvec_shfl_reduction(CSR_matrix d_mat, int M, const double *x, double *y)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row < M)
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