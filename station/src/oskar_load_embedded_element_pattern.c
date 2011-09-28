#include "station/oskar_load_embedded_element_pattern.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD 0.0174532925199432957692
#define round(x) ((x)>=0.0?(int)((x)+0.5):(int)((x)-0.5))

int oskar_load_embedded_element_pattern(const char* filename, oskar_AntennaData* data)
{
    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return 1;

    // Initialise the flags and local data.
    int n = 0;
    float inc_theta = 0.0f, inc_phi = 0.0f;
    float theta = 0.0f, phi = 0.0f, p_theta = 0.0f, p_phi = 0.0f;
    float abs_theta, phase_theta, abs_phi, phase_phi;
    float min_theta = FLT_MAX, max_theta = -FLT_MAX;
    float min_phi = FLT_MAX, max_phi = -FLT_MAX;

    // Read the first line and check if data is in logarithmic format.
    char line[1024];
    if (!fgets(line, sizeof(line), file)) return 1;
    const char* dbi = strstr(line, "dBi"); // Check for presence of "dBi".

    // Initialise pointers to NULL.
    data->g_phi = NULL;
    data->g_theta = NULL;

    // Loop over and read each line in the file.
    while (fgets(line, sizeof(line), file))
    {
        // Parse the line.
        int a = sscanf(line, "%f %f %*f %f %f %f %f %*f", &theta, &phi,
                    &abs_theta, &phase_theta, &abs_phi, &phase_phi);

        // Check that data was read correctly.
        if (a != 6) continue;

        // Ignore any data below horizon.
        if (theta > 90.0) continue;

        // Convert coordinates to radians.
        theta *= DEG2RAD;
        phi *= DEG2RAD;

        // Set coordinate increments.
        if (inc_theta <= FLT_EPSILON) inc_theta = theta - p_theta;
        if (inc_phi <= FLT_EPSILON) inc_phi = phi - p_phi;

        // Set ranges.
        if (theta < min_theta) min_theta = theta;
        if (theta > max_theta) max_theta = theta;
        if (phi < min_phi) min_phi = phi;
        if (phi > max_phi) max_phi = phi;

        // Ensure enough space in arrays.
        if (n % 100 == 0)
        {
            int size = (n + 100) * sizeof(float);
            data->g_theta = (float2*) realloc(data->g_theta, 2*size);
            data->g_phi   = (float2*) realloc(data->g_phi, 2*size);
        }

        // Store the coordinates in radians.
        p_theta = theta;
        p_phi = phi;

        // Convert decibel to linear scale if necessary.
        if (dbi)
        {
            abs_theta = pow(10.0, abs_theta / 10.0);
            abs_phi   = pow(10.0, abs_phi / 10.0);
        }

        // Amp,phase to real,imag conversion.
        data->g_theta[n].x = abs_theta * cos(phase_theta * DEG2RAD);
        data->g_theta[n].y = abs_theta * sin(phase_theta * DEG2RAD);
        data->g_phi[n].x = abs_phi * cos(phase_phi * DEG2RAD);
        data->g_phi[n].y = abs_phi * sin(phase_phi * DEG2RAD);

        // Increment array pointer.
        n++;
    }
    fclose(file);

    // Get number of points in each dimension.
    float n_theta = (max_theta - min_theta) / inc_theta;
    float n_phi = (max_phi - min_phi) / inc_phi;

    // Store number of points in arrays.
    data->n_points = n;
    data->n_theta = 1 + round(n_theta); // Must round to nearest integer.
    data->n_phi = 1 + round(n_phi); // Must round to nearest integer.
    data->min_theta = min_theta;
    data->min_phi = min_phi;
    data->max_theta = max_theta;
    data->max_phi = max_phi;
    data->inc_theta = inc_theta;
    data->inc_phi = inc_phi;

    return 0;
}

#ifdef __cplusplus
}
#endif
