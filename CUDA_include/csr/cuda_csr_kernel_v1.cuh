#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../../src/data_structures/csr_matrix.h"

/*********************************************************************************************************
 * Primo Kernel CUDA elaborato. Esso calcola il prodotto matrice-vettore (SpMV) in formato CSR.
 * Ogni thread elabora una riga della matrice: legge gli indici non nulli (JA),
 * moltiplica i valori corrispondenti (AS) con gli elementi del vettore x e accumula il risultato in y.
 * Ogni blocco elabora una riga della matrice: legge gli indici non nulli (JA), moltiplica i
 * valori corrispondenti (AS) con gli elementi del vettore x e accumula il risultato in y.
 *********************************************************************************************************/

__global__ void csr_matvec_kernel(int *d_IRP, int *d_JA, double *d_AS, int rows, double *d_x, double *d_y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        double sum = 0.0;
        int start = d_IRP[row];
        int end = d_IRP[row + 1];
        for (int i = start; i < end; i++)
        {
            int col = d_JA[i];
            double val = d_AS[i];
            sum += val * d_x[col];
        }
        d_y[row] = sum;
    }
}