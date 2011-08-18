#include "utility/oskar_load_csv_coordinates.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_load_csv_coordinates(const char* filename,
        unsigned* n, double** x, double** y)
{
    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;
    *n = 0;

    double ax, ay;
    while (fscanf(file, "%lf,%lf", &ax, &ay) != EOF)
    {
        // Ensure enough space in arrays.
        if (*n % 100 == 0)
        {
            int size = ((*n) + 100) * sizeof(double);
            *x = (double*) realloc(*x, size);
            *y = (double*) realloc(*y, size);
        }

        // Store the data.
        (*x)[*n] = ax;
        (*y)[*n] = ay;
        (*n)++;
    }
    fclose(file);

    return *n;
}

#ifdef __cplusplus
}
#endif
