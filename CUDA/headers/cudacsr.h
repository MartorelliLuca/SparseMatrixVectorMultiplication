#ifndef CSR_HEADERS_H
#define CSR_HEADERS_H

#include "../../src/headers/csr_headers.h"
#include "../src/data_structures/csr_matrix.h"

double call_kernel_v1(CSRMatrix *csr, double *x, double *y);
double call_kernel_v2(CSRMatrix *csr, double *x, double *y);
double call_kernel_v3(CSRMatrix *csr, double *x, double *y);

#endif
