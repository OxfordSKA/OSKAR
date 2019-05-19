/*
 * Copyright (c) 2019, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
        const char* filename, double freq_hz, int* status)
{
    int i, j, n, offset = -1, selector = 0, *l_max1 = 0, *l_max2 = 0;
    oskar_Mem *sw1 = 0, *sw2 = 0;
    char *line = 0;
    size_t bufsize = 0, num_tmp = 0;
    double *tmp_coeff = 0;
    FILE* file;
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
        if (fabs(data->freqs_hz[i] - freq_hz) <= freq_hz * DBL_EPSILON)
            break;
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
        selector |= OPTION_X;
    if (strstr(leafname, "_y_") || strstr(leafname, "_Y_"))
        selector |= OPTION_Y;
    if (strstr(leafname, "_te") || strstr(leafname, "_TE"))
        selector |= OPTION_TE;
    if (strstr(leafname, "_tm") || strstr(leafname, "_TM"))
        selector |= OPTION_TM;
    if (strstr(leafname, "_re") || strstr(leafname, "_RE"))
        offset = 0;
    if (strstr(leafname, "_im") || strstr(leafname, "_IM"))
        offset = 1;
    if (offset < 0)
    {
        oskar_log_error("Unknown spherical wave filename pattern '%s'",
                filename);
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Get pointer to spherical wave data using selector and frequency. */
    switch (selector)
    {
    case OPTION_TE:
        l_max1 = &data->x_lmax[i];
        l_max2 = &data->y_lmax[i];
        sw1 = data->x_te[i];
        sw2 = data->y_te[i];
        break;
    case OPTION_TM:
        l_max1 = &data->x_lmax[i];
        l_max2 = &data->y_lmax[i];
        sw1 = data->x_tm[i];
        sw2 = data->y_tm[i];
        break;
    case OPTION_X | OPTION_TE:
        l_max1 = &data->x_lmax[i];
        sw1 = data->x_te[i];
        break;
    case OPTION_X | OPTION_TM:
        l_max1 = &data->x_lmax[i];
        sw1 = data->x_tm[i];
        break;
    case OPTION_Y | OPTION_TE:
        l_max1 = &data->y_lmax[i];
        sw1 = data->y_te[i];
        break;
    case OPTION_Y | OPTION_TM:
        l_max1 = &data->y_lmax[i];
        sw1 = data->y_tm[i];
        break;
    default:
        oskar_log_error("Unknown spherical wave filename pattern '%s'",
                filename);
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over and read each line in the file. */
    const int type = oskar_element_precision(data);
    n = 0;
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        const int num_read = (int)oskar_string_to_array_realloc_d(
                line, &num_tmp, &tmp_coeff);
        if (sw1 && (int)oskar_mem_length(sw1) < n + num_read)
            oskar_mem_realloc(sw1, n + 10 * num_read, status);
        if (sw2 && (int)oskar_mem_length(sw2) < n + num_read)
            oskar_mem_realloc(sw2, n + 10 * num_read, status);
        if (*status) break;
        if (type == OSKAR_DOUBLE)
        {
            double* c = oskar_mem_double(sw1, status) + offset;
            for (j = 0; j < num_read; ++j)
                c[2 * (j + n)] = tmp_coeff[j];
            if (sw2)
            {
                double* c = oskar_mem_double(sw2, status) + offset;
                for (j = 0; j < num_read; ++j)
                    c[2 * (j + n)] = tmp_coeff[j];
            }
        }
        else
        {
            float* c = oskar_mem_float(sw1, status) + offset;
            for (j = 0; j < num_read; ++j)
                c[2 * (j + n)] = (float)(tmp_coeff[j]);
            if (sw2)
            {
                float* c = oskar_mem_float(sw2, status) + offset;
                for (j = 0; j < num_read; ++j)
                    c[2 * (j + n)] = (float)(tmp_coeff[j]);
            }
        }
        n += num_read;
    }

    /* Check total number of coefficients to find l_max. */
    const double l_max_d = (sqrt(1 + 8 * n) - 1) / 4;
    const int l_max = (int)round(l_max_d);
    if (fabs((double)l_max - l_max_d) < FLT_EPSILON)
    {
        /* Store the filename. */
        if (l_max1) *l_max1 = l_max;
        if (l_max2) *l_max2 = l_max;
        const size_t fname_len = 1 + strlen(filename);
        if (selector & OPTION_X)
            oskar_mem_append_raw(data->filename_x[i], filename,
                    OSKAR_CHAR, OSKAR_CPU, fname_len, status);
        else if (selector & OPTION_Y)
            oskar_mem_append_raw(data->filename_y[i], filename,
                    OSKAR_CHAR, OSKAR_CPU, fname_len, status);
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
        oskar_log_error("Number of spherical wave coefficients loaded "
                "(%d) does not match number required (%d) for closest value "
                "of l_max=%d in '%s'",
                n, (2 * l_max + 1) * l_max, l_max, filename);
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Free memory. */
    free(tmp_coeff);
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
