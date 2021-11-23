/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/oskar_element.h"
#include "telescope/station/element/private_element.h"

#include "math/oskar_cmath.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_load(oskar_Element* element,
        const char* filename, int x_pol, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = 0;
    size_t bufsize = 0;
    FILE* file = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Declare parameters. */
        char *list[6], base_type = 'D', taper_type = 'N', length_unit = 'W';
        double length = 0.5, cosine_power = 1.0;
        double gaussian_fwhm_deg = 45.0, gaussian_ref_freq_hz = 0.0;
        size_t num_read = 0, num_par = 6;

        /* Load element data. */
        num_read = oskar_string_to_array_s(line, num_par, list);
        if (num_read == 0) continue;
        base_type = toupper(list[0][0]);
        if (base_type == 'D' && num_read >= 3)
        {
            length = strtod(list[1], 0);
            length_unit = toupper(list[2][0]);
            if (num_read >= 4)
            {
                taper_type = toupper(list[3][0]);
                if (taper_type == 'C' && num_read >= 5)
                {
                    cosine_power = strtod(list[4], 0);
                }
                else if (taper_type == 'G' && num_read >= 6)
                {
                    gaussian_fwhm_deg = strtod(list[4], 0);
                    gaussian_ref_freq_hz = strtod(list[5], 0);
                }
                else continue;
            }
        }
        else if (base_type == 'I')
        {
            if (num_read >= 2)
            {
                taper_type = toupper(list[1][0]);
                if (taper_type == 'C' && num_read >= 3)
                {
                    cosine_power = strtod(list[2], 0);
                }
                else if (taper_type == 'G' && num_read >= 4)
                {
                    gaussian_fwhm_deg = strtod(list[2], 0);
                    gaussian_ref_freq_hz = strtod(list[3], 0);
                }
                else continue;
            }
        }
        else if (base_type == 'G')
        {
            if (num_read >= 2)
            {
                taper_type = toupper(list[1][0]);
                if (taper_type == 'C' && num_read >= 3)
                {
                    cosine_power = strtod(list[2], 0);
                }
                else if (taper_type == 'G' && num_read >= 4)
                {
                    gaussian_fwhm_deg = strtod(list[2], 0);
                    gaussian_ref_freq_hz = strtod(list[3], 0);
                }
                else continue;
            }
        }
        else continue;

        if (x_pol)
        {
            element->x_element_type = base_type;
            element->x_taper_type = taper_type;
            element->x_dipole_length = length;
            element->x_dipole_length_units = length_unit;
            element->x_taper_cosine_power = cosine_power;
            element->x_taper_gaussian_fwhm_rad = gaussian_fwhm_deg *
                    M_PI / 180.0;
            element->x_taper_ref_freq_hz = gaussian_ref_freq_hz;
        }
        else
        {
            element->y_element_type = base_type;
            element->y_taper_type = taper_type;
            element->y_dipole_length = length;
            element->y_dipole_length_units = length_unit;
            element->y_taper_cosine_power = cosine_power;
            element->y_taper_gaussian_fwhm_rad = gaussian_fwhm_deg *
                    M_PI / 180.0;
            element->y_taper_ref_freq_hz = gaussian_ref_freq_hz;
        }
    }

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
