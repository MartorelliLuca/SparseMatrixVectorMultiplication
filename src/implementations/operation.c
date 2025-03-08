#include <ctype.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bits/time.h>
#include <errno.h>

#include "../data_structures/csr_matix.h"
#include "../data_structures/ellpack_matrix.h"

// First attempt to do matrix-vector dot product in CSR format
void matvec_csr(CSR_matrix *csr_matrix, double *x, double *y)
{
#pragma omp parallel for
    for (int i = 0; i < csr_matrix->M; i++)
    {
        for (int j = csr_matrix->IRP[i]; j < csr_matrix->IRP[i + 1]; j++)
        {
            y[i] += csr_matrix->AS[j] * x[csr_matrix->JA[j]];
        }
    }
}

// First attempt to do matrix-vector dot produt in ELLPACK format
void matvec_ellpack(ELLPACK_matrix *ellpack_matrix, double *x, double *y)
{
}