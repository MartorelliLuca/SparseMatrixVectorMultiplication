#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>

#include "../CUDA_include/cudacsr.h"
#include "../src/data_structures/csr_matrix.h"
#include "../src/data_structures/performance.h"
#include "../CUDA_include/cuda_csr_kernel_v1.cuh"

// qua alloca strutture dati per le chiamate a kernel (partizione del carico di ogni warp)

double call_kernel_v1(CSR_matrix *csr, double *x, double *y)
{
    double *d_x, *d_y;
    int *d_row, *d_col;
    double *d_val;
    double *d_res;
    double res;
    cudaMalloc(&d_x, csr->N * sizeof(double));
    cudaMalloc(&d_y, csr->N * sizeof(double));
    cudaMalloc(&d_row, (csr->N + 1) * sizeof(int));
    cudaMalloc(&d_col, csr->non_zero_values * sizeof(int));
    cudaMalloc(&d_val, csr->non_zero_values * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    if (d_x == NULL || d_y == NULL || d_row == NULL || d_col == NULL || d_val == NULL || d_res == NULL)
    {
        printf("Errore nell'allocazione della memoria per il device\n");
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_x, x, csr->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, csr->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, csr->IRP, (csr->N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr->JA, csr->non_zero_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, csr->AS, csr->non_zero_values * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (csr->N + block_size - 1) / block_size;
    csr_matvec_warps_per_row<<<num_blocks, block_size>>>(csr->N, d_row, d_col, d_val, d_x, d_y, d_res);
    // mi permette di sincronizzare il device con l'host
    cudaDeviceSynchronize();
    // copia risultato da device
    cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_res);

    return res;
}