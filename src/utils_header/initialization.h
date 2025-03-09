#ifndef INIZIALITAZION_H
#define INIZIALITAZION_H

FILE *get_matrix_file(char *dir_name, char *matrix_filename, int *file_type);
double *initialize_x_vector(int size);
double *initialize_y_vector(int size);

#endif