#ifndef MATRIX_H
#define MATRIX_H

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
    //"amazon0302.mtx",
    "af_1_k101.mtx", // ok
    //"roadNet-PA.mtx"
};

typedef struct
{
    int M;                // Row Number
    int N;                // Colum Number
    int NZ;               // Non-Zero Number
    int *row_indices;     // Array of row indices of non zeroes values
    int *columns_indices; // Array of column indices of non zeroes values
    double *values;
} matrix_format;

static const int num_matrices = sizeof(matrix_filenames) / sizeof(matrix_filenames[0]);

#endif