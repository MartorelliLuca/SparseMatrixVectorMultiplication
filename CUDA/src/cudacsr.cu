#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include "../headers/cudacsr.h"
#include "../kernel/csr/cudakernel1.cuh"

// qua alloca strutture dati per le chiamate a kernel (partizione del carico di ogni warp)

double call_kernel_v1(CSRMatrix *csr, double *x, double *y)
{

    double *d_x, *d_y;
    int *d_row, *d_col;
    double *d_val;
    double *d_res;
    double res;
    cudaMalloc(&d_x, csr->n * sizeof(double));
    cudaMalloc(&d_y, csr->n * sizeof(double));
    cudaMalloc(&d_row, (csr->n + 1) * sizeof(int));
    cudaMalloc(&d_col, csr->nnz * sizeof(int));
    cudaMalloc(&d_val, csr->nnz * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    if (d_x == NULL || d_y == NULL || d_row == NULL || d_col == NULL || d_val == NULL || d_res == NULL)
    {
        printf("Errore nell'allocazione della memoria per il device\n");
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_x, x, csr->n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, csr->n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, csr->row, (csr->n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr->col, csr->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, csr->val, csr->nnz * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (csr->n + block_size - 1) / block_size;
    csr_matvec_warps_per_row<<<num_blocks, block_size>>>(csr->n, d_row, d_col, d_val, d_x, d_y, d_res);
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