#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include "data_structures/ellpack_matrix.h"
#include "data_structures/csr_matix.h"

const char *dir_name = "matrici";
DIR *dir;
struct dirent *entry;

int main()
{
    // prendi matrice 1 dalla cartella matrici e forma ellpack e csr attraverso read e write
    dir = opendir(dir_name);
    if (dir == NULL)
    {
        perror("Errore nell'apertura della cartella");
        return 1;
    }
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_name[0] == '.')
            continue;
        // fai roba
        printf("File: %s\n", entry->d_name);
        sleep(1);
    }

    closedir(dir);
    return 0;
}