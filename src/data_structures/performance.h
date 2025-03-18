#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include "../utils_header/computation_type.h"

struct performance
{
    char matrix[256];              // Matrix Name
    int number_of_threads_used;    // Number of threads used
    computation_time computation;  // Type of computation
    int non_zeroes_values;         // Number of Non zeroes value
    double time_used;              // Time used to dot-product operation
    double flops;                  // FLOPS
    double mflops;                 // MEGAFLOPS
    double gflops;                 // GIGAFLOPS
    struct performance *next_node; // Pointer to Next performance node
    struct performance *prev_node; // Pointer to Prev performance node
};

#endif