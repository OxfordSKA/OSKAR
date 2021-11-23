/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OPTION_X      = 1 << 1,
    OPTION_Y      = 1 << 2,
    OPTION_TE     = 1 << 3,
    OPTION_TM     = 1 << 4
};

void oskar_element_load_spherical_wave_coeff(oskar_Element* data,
        const char* filename, double freq_hz, int* num_tmp, double** tmp,
        int* status)
{
    int i = 0, j = 0, n = 0, selector = 0, line_counter = 1;
    int offset_complex = -1, offset1 = 0, offset2 = -1;
    oskar_Mem* sw = 0;
    void* sw_p = 0;
    char *line = 0;
    size_t bufsize = 0;
    FILE* file = 0;
    if (*status) return;

    /* Check the location. */
    if (oskar_element_mem_location(data) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check if this frequency has already been set, and get its index if so. */
    n = data->num_freq;
    for (i = 0; i < n; ++i)
    {
        if (fabs(data->freqs_hz[i] - freq_hz) <= freq_hz * DBL_EPSILON) break;
    }

    /* Expand arrays to hold data for a new frequency, if needed. */
    if (i >= data->num_freq)
    {
        i = data->num_freq;
        oskar_element_resize_freq_data(data, i + 1, status);
        data->freqs_hz[i] = freq_hz;
    }

    /* Get leafname of file and parse it. */
    const char* leafname = oskar_dir_leafname(filename);
    if (strstr(leafname, "_x_") || strstr(leafname, "_X_"))
    {
        selector |= OPTION_X;
    }
    if (strstr(leafname, "_y_") || strstr(leafname, "_Y_"))
    {
        selector |= OPTION_Y;
    }
    if (strstr(leafname, "_te") || strstr(leafname, "_TE"))
    {
        selector |= OPTION_TE;
    }
    if (strstr(leafname, "_tm") || strstr(leafname, "_TM"))
    {
        selector |= OPTION_TM;
    }
    if (strstr(leafname, "_re") || strstr(leafname, "_RE"))
    {
        offset_complex = 0;
    }
    if (strstr(leafname, "_im") || strstr(leafname, "_IM"))
    {
        offset_complex = 1;
    }
    if (offset_complex < 0)
    {
        oskar_log_error(0, "Unknown spherical wave filename pattern '%s'",
                filename);
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Get pointer to spherical wave data using selector and frequency. */
    /* Coefficient data stored as:
     * (x_te_re, x_te_im), (x_tm_re, x_tm_im),
     * (y_te_re, y_te_im), (y_tm_re, y_tm_im)   */
    /* Separate patterns in each polarisation are already rotated,
     * so use common phi coordinates when evaluating them. */
    switch (selector)
    {
    case OPTION_TE:
        offset1 = 0 + offset_complex;
        offset2 = 4 + offset_complex;
        data->common_phi_coords[i] = 0;
        break;
    case OPTION_TM:
        offset1 = 2 + offset_complex;
        offset2 = 6 + offset_complex;
        data->common_phi_coords[i] = 0;
        break;
    case OPTION_X | OPTION_TE:
        offset1 = 0 + offset_complex;
        data->common_phi_coords[i] = 1;
        break;
    case OPTION_X | OPTION_TM:
        offset1 = 2 + offset_complex;
        data->common_phi_coords[i] = 1;
        break;
    case OPTION_Y | OPTION_TE:
        offset1 = 4 + offset_complex;
        data->common_phi_coords[i] = 1;
        break;
    case OPTION_Y | OPTION_TM:
        offset1 = 6 + offset_complex;
        data->common_phi_coords[i] = 1;
        break;
    default:
        oskar_log_error(0, "Unknown spherical wave filename pattern '%s'",
                filename);
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    const int type = data->precision;
    if (!data->sph_wave[i])
    {
        data->sph_wave[i] = oskar_mem_create(
                type | OSKAR_COMPLEX | OSKAR_MATRIX, OSKAR_CPU, 0, status);
    }
    sw = data->sph_wave[i];
    sw_p = oskar_mem_void(sw);

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over and read each line in the file. */
    n = 0;
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        const int num_m = (2 * line_counter + 1);
        if (*num_tmp < n + num_m)
        {
            *num_tmp += num_m;
            *tmp = (double*) realloc(*tmp, *num_tmp * sizeof(double));
        }
        const int num_read = (int)oskar_string_to_array_d(
                line, num_m, *tmp + n);
        if (num_read < num_m) continue;
        n += num_m;
        line_counter++;
    }
    if ((int)oskar_mem_length(sw) < n)
    {
        oskar_mem_realloc(sw, (size_t)n, status);
        sw_p = oskar_mem_void(sw);
        if (*status) return;
    }
    if (type == OSKAR_DOUBLE)
    {
        double* c1 = ((double*)sw_p) + offset1;
        if (offset2 >= 0)
        {
            double* c2 = ((double*)sw_p) + offset2;
            for (j = 0; j < n; ++j)
            {
                c1[8 * j] = (*tmp)[j];
                c2[8 * j] = (*tmp)[j];
            }
        }
        else
        {
            for (j = 0; j < n; ++j)
            {
                c1[8 * j] = (*tmp)[j];
            }
        }
    }
    else
    {
        float* c1 = ((float*)sw_p) + offset1;
        if (offset2 >= 0)
        {
            float* c2 = ((float*)sw_p) + offset2;
            for (j = 0; j < n; ++j)
            {
                float t = (float)((*tmp)[j]);
                c1[8 * j] = t;
                c2[8 * j] = t;
            }
        }
        else
        {
            for (j = 0; j < n; ++j)
            {
                c1[8 * j] = (float)((*tmp)[j]);
            }
        }
    }

    /* Check total number of coefficients to find l_max. */
    const double l_max_d = sqrt(n + 1) - 1;
    const int l_max = (int)round(l_max_d);
    if (fabs((double)l_max - l_max_d) < FLT_EPSILON)
    {
        /* Store the filename. */
        if (data->l_max[i] == 0 || data->l_max[i] == l_max)
        {
            data->l_max[i] = l_max;
            const size_t fname_len = 1 + strlen(filename);
            if (!data->filename_x[i])
            {
                data->filename_x[i] = oskar_mem_create(
                        OSKAR_CHAR, OSKAR_CPU, fname_len, status);
            }
            if (!data->filename_y[i])
            {
                data->filename_y[i] = oskar_mem_create(
                        OSKAR_CHAR, OSKAR_CPU, fname_len, status);
            }
            if (selector & OPTION_X)
            {
                oskar_mem_append_raw(data->filename_x[i], filename,
                        OSKAR_CHAR, OSKAR_CPU, fname_len, status);
            }
            else if (selector & OPTION_Y)
            {
                oskar_mem_append_raw(data->filename_y[i], filename,
                        OSKAR_CHAR, OSKAR_CPU, fname_len, status);
            }
            else
            {
                oskar_mem_append_raw(data->filename_x[i], filename,
                        OSKAR_CHAR, OSKAR_CPU, fname_len, status);
                oskar_mem_append_raw(data->filename_y[i], filename,
                        OSKAR_CHAR, OSKAR_CPU, fname_len, status);
            }
        }
        else
        {
            oskar_log_error(0, "Inconsistent number of coefficients for "
                    "spherical wave pattern (current=%d, expected=%d) in '%s",
                    l_max, data->l_max[i], filename);
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
        }
    }
    else
    {
        oskar_log_error(0, "Number of spherical wave coefficients loaded "
                "(%d) does not match number required (%d) for closest value "
                "of l_max=%d in '%s'",
                n, (l_max + 1) * (l_max + 1) - 1, l_max, filename);
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Free memory. */
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
