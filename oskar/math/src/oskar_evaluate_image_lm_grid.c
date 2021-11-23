/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_evaluate_image_lm_grid.h"
#include "math/oskar_linspace.h"

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_image_lm_grid(int num_l, int num_m, double fov_lon,
        double fov_lat, oskar_Mem* l, oskar_Mem* m, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(l);
    const int loc = oskar_mem_location(l);
    if (oskar_mem_type(m) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_location(m) != loc)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (loc == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_image_lm_grid_d(num_l, num_m,
                    fov_lon, fov_lat,
                    oskar_mem_double(l, status),
                    oskar_mem_double(m, status));
        }
        else if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_image_lm_grid_f(num_l, num_m,
                    (float)fov_lon, (float)fov_lat,
                    oskar_mem_float(l, status),
                    oskar_mem_float(m, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        /* There is currently no need for a GPU version of this function */
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
}

/* Single precision. */
void oskar_evaluate_image_lm_grid_f(int num_l, int num_m,
        float fov_lon, float fov_lat, float* grid_l, float* grid_m)
{
    int i = 0, j = 0, p = 0;
    float l_max = 0.0f, m_max = 0.0f, *l = 0, *m = 0, r = 0.0f;

    /* Allocate temporary memory for vectors. */
    l = (float*) malloc(num_l * sizeof(float));
    m = (float*) malloc(num_m * sizeof(float));

    /* Set up the grid boundaries. */
    l_max = sin(0.5 * fov_lon);
    m_max = sin(0.5 * fov_lat);

    /* Create the axis vectors. */
    oskar_linspace_f(l, l_max, -l_max, num_l); /* FITS convention. */
    oskar_linspace_f(m, -m_max, m_max, num_m);

    /* Slowest varying is m, fastest varying is l. */
    for (j = 0, p = 0; j < num_m; ++j)
    {
        for (i = 0; i < num_l; ++i, ++p)
        {
            r = sqrtf(l[i]*l[i] + m[j]*m[j]);
            if (r > 1.0f)
            {
                grid_l[p] = sqrtf(-1.0f); /* NAN */
                grid_m[p] = sqrtf(-1.0f); /* NAN */
            }
            else
            {
                grid_l[p] = l[i];
                grid_m[p] = m[j];
            }
        }
    }

    /* Free temporary memory. */
    free(l);
    free(m);
}

/* Double precision. */
void oskar_evaluate_image_lm_grid_d(int num_l, int num_m,
        double fov_lon, double fov_lat, double* grid_l, double* grid_m)
{
    int i = 0, j = 0, p = 0;
    double l_max = 0.0, m_max = 0.0, *l = 0, *m = 0, r = 0.0;

    /* Allocate temporary memory for vectors. */
    l = (double*) malloc(num_l * sizeof(double));
    m = (double*) malloc(num_m * sizeof(double));

    /* Set up the grid boundaries. */
    l_max = sin(0.5 * fov_lon);
    m_max = sin(0.5 * fov_lat);

    /* Create the axis vectors. */
    oskar_linspace_d(l, l_max, -l_max, num_l); /* FITS convention. */
    oskar_linspace_d(m, -m_max, m_max, num_m);

    /* Slowest varying is m, fastest varying is l. */
    for (j = 0, p = 0; j < num_m; ++j)
    {
        for (i = 0; i < num_l; ++i, ++p)
        {
            r = sqrt(l[i]*l[i] + m[j]*m[j]);
            if (r > 1.0)
            {
                grid_l[p] = sqrtf(-1.0f); /* NAN */
                grid_m[p] = sqrtf(-1.0f); /* NAN */
            }
            else
            {
                grid_l[p] = l[i];
                grid_m[p] = m[j];
            }
        }
    }

    /* Free temporary memory. */
    free(l);
    free(m);
}

#ifdef __cplusplus
}
#endif
