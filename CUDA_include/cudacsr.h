#ifndef CUDACSR_H
#define CUDACSR_H

#include "../src/headers/csr_headers.h"
#include "../src/data_structures/csr_matrix.h"

double call_kernel_v1(CSR_matrix *csr, double *x, double *y);
double call_kernel_v2(CSR_matrix *csr, double *x, double *y);
double call_kernel_v3(CSR_matrix *csr, double *x, double *y);
void callcsr();

#endif
