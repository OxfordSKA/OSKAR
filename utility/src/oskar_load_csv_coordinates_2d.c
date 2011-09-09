#include "utility/oskar_load_csv_coordinates_2d.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_load_csv_coordinates_2d_d(const char* filename, unsigned* n,
        double** x, double** y)
{
    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;
    *n = 0;

    *x = NULL;
    *y = NULL;

    double ax, ay;
    while (fscanf(file, "%lf,%lf", &ax, &ay) != EOF)
    {
        // Ensure enough space in arrays.
        if (*n % 100 == 0)
        {
            size_t mem_size = ((*n) + 100) * sizeof(double);
            *x = (double*) realloc(*x, mem_size);
            *y = (double*) realloc(*y, mem_size);
        }

        // Store the data.
        (*x)[*n] = ax;
        (*y)[*n] = ay;
        (*n)++;
    }
    fclose(file);

    return *n;
}


int oskar_load_csv_coordinates_2d_f(const char* filename, unsigned* n,
        float** x, float** y)
{
    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 0;
    *n = 0;

    *x = NULL;
    *y = NULL;

    float ax, ay;
    while (fscanf(file, "%f,%f", &ax, &ay) != EOF)
    {
        // Ensure enough space in arrays.
        if (*n % 100 == 0)
        {
            size_t mem_size = ((*n) + 100) * sizeof(float);
            *x = (float*) realloc(*x, mem_size);
            *y = (float*) realloc(*y, mem_size);
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
