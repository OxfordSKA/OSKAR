#include "utility/oskar_load_csv_coordinates.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

int oskar_load_csv_coordinates(const char* filename,
        unsigned* n, double** x, double** y)
{
    FILE * pFile;
    pFile = fopen(filename, "r");
    if (pFile == NULL) return 0;

    vector<double> temp_x;
    vector<double> temp_y;
    int num_antennas = 0;

    float ax, ay;
    while(fscanf(pFile, "%f,%f", &ax, &ay) != EOF)
    {
        temp_x.push_back(ax);
        temp_y.push_back(ay);
        num_antennas++;
    }
    fclose(pFile);

    size_t mem_size = num_antennas * sizeof(double);
    *n = num_antennas;
    *x = (double*) malloc(mem_size);
    memcpy((void*)*x, (const void*)&temp_x[0], mem_size);
    *y = (double*) malloc(mem_size);
    memcpy((void*)*y, (const void*)&temp_y[0], mem_size);

    return num_antennas;
}
