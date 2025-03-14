#ifndef MATRIX_FORMAT_H
#define MATRIX_FORMAT_H

#include "../utils_header/mmio.h"

static const char *matrix_filenames[] = {
    "cage4.mtx",           // ok
    "mhda416.mtx",         // ok
    "mcfe.mtx",            // ok
    "olm1000.mtx",         // ok
    "adder_dcop_32.mtx",   // ok
    "west2021.mtx",        // ok
    "cavity10.mtx",        // ok
    "rdist2.mtx",          // ok
    "cant.mtx",            // ok
    "olafu.mtx",           // ok
    "Cube_Coup_dt0.mtx",   // ok
    "ML_Laplace.mtx",      // ok
    "bcsstk17.mtx",        // ok
    "mac_econ_fwd500.mtx", // ok
    "mhd4800a.mtx",        // ok
    "cop20k_A.mtx",        // ok
    "raefsky2.mtx",        // ok
    "af23560.mtx",         // ok
    "lung2.mtx",           // ok
    "PR02R.mtx",           // ok
    "FEM_3D_thermal1.mtx", // ok
    "thermal1.mtx",        // ok
    "thermal2.mtx",        // ok
    "thermomech_TK.mtx",   // ok
    "nlpkkt80.mtx",        // ok
    "webbase-1M.mtx",      // ok
    "dc1.mtx",             // ok
    "amazon0302.mtx",
    "af_1_k101.mtx", // ok
    "roadNet-PA.mtx"};

typedef struct
{
    char name[256];                   // Matrix Name
    int *rows;                        // Row indices
    int *columns;                     // Columns indices
    double *values;                   // Non zeroes values
    int M;                            // Rows
    int N;                            // Columns
    int number_of_non_zeoroes_values; // Number of nonzeroes values
    MM_typecode matrix_typecode;      // Matrix typecode

} matrix_format;

static const int num_matrices = sizeof(matrix_filenames) / sizeof(matrix_filenames[0]);

#endif