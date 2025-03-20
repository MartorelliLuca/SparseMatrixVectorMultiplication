#ifndef CSR_HEADERS_H
#define CSR_HEADERS_H

#include "/src/utils_header/csr_headers.h"
#include "/src/data_structures/csr_matrix.h"
#include "/src/header/matrix_format.h"

double csr_matvec_thread(CSRMatrix *csr, double *x, double *y);
double csr_matvec_warps_per_row(CSRMatrix *csr, double *x, double *y, int warps_per_row);

#endif