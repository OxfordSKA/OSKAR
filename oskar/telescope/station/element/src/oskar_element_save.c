/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"
#include "math/oskar_cmath.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_save(const oskar_Element* element, const char* filename,
        int x_pol, int* status)
{
    FILE* file;
    char base_type, taper_type;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Save the data. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Get the base type and the taper type. */
    base_type  = x_pol ? element->x_element_type : element->y_element_type;
    taper_type = x_pol ? element->x_taper_type : element->y_taper_type;

    /* Switch on base type. */
    if (base_type == 'I')
    {
        fprintf(file, "I ");
    }
    else if (base_type == 'D')
    {
        char length_units;
        double length;
        length = x_pol ? element->x_dipole_length : element->y_dipole_length;
        length_units = x_pol ? element->x_dipole_length_units :
                element->y_dipole_length_units;
        fprintf(file, "D %.14e %c ", length, length_units);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        fclose(file);
        return;
    }

    /* Switch on taper type. */
    if (taper_type == 'C')
    {
        double cosine_power;
        cosine_power = x_pol ? element->x_taper_cosine_power :
                element->y_taper_cosine_power;
        fprintf(file, "C %.14e\n", cosine_power);
    }
    else if (taper_type == 'G')
    {
        double fwhm_deg, ref_freq_hz;
        fwhm_deg = (x_pol ? element->x_taper_gaussian_fwhm_rad :
                element->y_taper_gaussian_fwhm_rad) * 180.0 / M_PI;
        ref_freq_hz = x_pol ? element->x_taper_ref_freq_hz:
                element->y_taper_ref_freq_hz;
        fprintf(file, "G %.14e %.14e\n", fwhm_deg, ref_freq_hz);
    }
    else
    {
        fprintf(file, "N\n");
    }

    fclose(file);
}

#ifdef __cplusplus
}
#endif
