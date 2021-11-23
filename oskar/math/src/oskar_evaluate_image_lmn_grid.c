/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_fov_to_cellsize.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "math/oskar_linspace.h"

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper */
void oskar_evaluate_image_lmn_grid(int num_l, int num_m, double fov_lon,
        double fov_lat, int centred, oskar_Mem* grid_l, oskar_Mem* grid_m,
        oskar_Mem* grid_n, int* status)
{
    if (*status) return;

    /* Check data type and location consistency. */
    const int type = oskar_mem_type(grid_l);
    const int loc = oskar_mem_location(grid_l);
    if (oskar_mem_type(grid_m) != type || oskar_mem_type(grid_n) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_location(grid_m) != loc || oskar_mem_location(grid_n) != loc)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* There is currently no need for a GPU version of this function. */
    if (loc != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    if (type == OSKAR_DOUBLE)
    {
        oskar_evaluate_image_lmn_grid_d(num_l, num_m, fov_lon,
                fov_lat, centred, oskar_mem_double(grid_l, status),
                oskar_mem_double(grid_m, status),
                oskar_mem_double(grid_n, status));
    }
    else if (type == OSKAR_SINGLE)
    {
        oskar_evaluate_image_lmn_grid_f(num_l, num_m, (float)fov_lon,
                (float)fov_lat, centred, oskar_mem_float(grid_l, status),
                oskar_mem_float(grid_m, status),
                oskar_mem_float(grid_n, status));
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}


/* Single precision. */
void oskar_evaluate_image_lmn_grid_f(int num_l, int num_m, float fov_lon,
        float fov_lat, int centred, float* grid_l, float* grid_m,
        float* grid_n)
{
    int i = 0, j = 0, p = 0;

    if (centred)
    {
        double l_max = 0.0, m_max = 0.0, *l = 0, *m = 0, r = 0.0;

        /* Allocate temporary memory for vectors. */
        l = (double*) malloc(num_l * sizeof(double));
        m = (double*) malloc(num_m * sizeof(double));

        /* Set up the grid boundaries. */
        l_max = -sin(0.5 * fov_lon); /* FITS convention. */
        m_max =  sin(0.5 * fov_lat);

        /* Create the axis vectors. */
        oskar_linspace_d(l, -l_max, l_max, num_l);
        oskar_linspace_d(m, -m_max, m_max, num_m);

        /* Slowest varying is m, fastest varying is l. */
        for (j = 0, p = 0; j < num_m; ++j)
        {
            for (i = 0; i < num_l; ++i, ++p)
            {
                r = sqrt(l[i]*l[i] + m[j]*m[j]);
                if (r > 1.0f)
                {
                    grid_l[p] = sqrt(-1.0); /* NAN */
                    grid_m[p] = sqrt(-1.0); /* NAN */
                    grid_n[p] = sqrt(-1.0); /* NAN */
                }
                else
                {
                    grid_l[p] = l[i];
                    grid_m[p] = m[j];
                    grid_n[p] = sqrt(1.0 - (grid_l[p]*grid_l[p])
                            - (grid_m[p]*grid_m[p]));
                }
            }
        }

        /* Free temporary memory. */
        free(l);
        free(m);
    }
    else
    {
        double l = 0.0, m = 0.0, delta_l = 0.0, delta_m = 0.0, r = 0.0;

        delta_l = sin(oskar_convert_fov_to_cellsize(fov_lon, num_l));
        delta_m = sin(oskar_convert_fov_to_cellsize(fov_lat, num_m));

        /* Slowest varying is m, fastest varying is l. */
        for (j = 0, p = 0; j < num_m; ++j)
        {
            m = (-(num_m / 2) + j) * delta_m;
            for (i = 0; i < num_l; ++i, ++p)
            {
                l = ((num_l / 2) - i) * delta_l;
                r = sqrt(l*l + m*m);
                if (r > 1.0)
                {
                    grid_l[p] = sqrt(-1.0); /* NAN */
                    grid_m[p] = sqrt(-1.0); /* NAN */
                    grid_n[p] = sqrt(-1.0); /* NAN */
                }
                else
                {
                    grid_l[p] = l;
                    grid_m[p] = m;
                    grid_n[p] = sqrt(1.0 - l*l - m*m);
                }
            }
        }
    }
}


/* Double precision. */
void oskar_evaluate_image_lmn_grid_d(int num_l, int num_m, double fov_lon,
        double fov_lat, int centred, double* grid_l, double* grid_m,
        double* grid_n)
{
    int i = 0, j = 0, p = 0;

    if (centred)
    {
        double l_max = 0.0, m_max = 0.0, *l = 0, *m = 0, r = 0.0;

        /* Allocate temporary memory for vectors. */
        l = (double*) malloc(num_l * sizeof(double));
        m = (double*) malloc(num_m * sizeof(double));

        /* Set up the grid boundaries. */
        l_max = -sin(0.5 * fov_lon); /* FITS convention. */
        m_max =  sin(0.5 * fov_lat);

        /* Create the axis vectors. */
        oskar_linspace_d(l, -l_max, l_max, num_l);
        oskar_linspace_d(m, -m_max, m_max, num_m);

        /* Slowest varying is m, fastest varying is l. */
        for (j = 0, p = 0; j < num_m; ++j)
        {
            for (i = 0; i < num_l; ++i, ++p)
            {
                r = sqrt(l[i]*l[i] + m[j]*m[j]);
                if (r > 1.0)
                {
                    grid_l[p] = sqrt(-1.0); /* NAN */
                    grid_m[p] = sqrt(-1.0); /* NAN */
                    grid_n[p] = sqrt(-1.0); /* NAN */
                }
                else
                {
                    grid_l[p] = l[i];
                    grid_m[p] = m[j];
                    grid_n[p] = sqrt(1.0 - (grid_l[p]*grid_l[p])
                            - (grid_m[p]*grid_m[p]));
                }
            }
        }

        /* Free temporary memory. */
        free(l);
        free(m);
    }
    else
    {
        double l = 0.0, m = 0.0, delta_l = 0.0, delta_m = 0.0, r = 0.0;

        delta_l = sin(oskar_convert_fov_to_cellsize(fov_lon, num_l));
        delta_m = sin(oskar_convert_fov_to_cellsize(fov_lat, num_m));

        /* Slowest varying is m, fastest varying is l. */
        for (j = 0, p = 0; j < num_m; ++j)
        {
            m = (-(num_m / 2) + j) * delta_m;
            for (i = 0; i < num_l; ++i, ++p)
            {
                l = ((num_l / 2) - i) * delta_l;
                r = sqrt(l*l + m*m);
                if (r > 1.0)
                {
                    grid_l[p] = sqrt(-1.0); /* NAN */
                    grid_m[p] = sqrt(-1.0); /* NAN */
                    grid_n[p] = sqrt(-1.0); /* NAN */
                }
                else
                {
                    grid_l[p] = l;
                    grid_m[p] = m;
                    grid_n[p] = sqrt(1.0 - l*l - m*m);
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
