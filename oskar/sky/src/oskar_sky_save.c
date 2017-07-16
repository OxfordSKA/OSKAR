/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "sky/oskar_sky.h"
#include "math/oskar_cmath.h"
#include <stdio.h>

#define RAD2DEG 180.0/M_PI
#define RAD2ARCSEC RAD2DEG * 3600.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_save(const char* filename, const oskar_Sky* sky, int* status)
{
    int s, type, num_sources;
    FILE* file;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check sky model is in CPU memory. */
    if (oskar_sky_mem_location(sky) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Open the output file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Get the data type and number of sources. */
    type = oskar_sky_precision(sky);
    num_sources = oskar_sky_num_sources(sky);

    /* Print a helpful header. */
    fprintf(file, "# Number of sources: %i\n", num_sources);
    fprintf(file, "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), "
            "Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2), "
            "FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)\n");

    /* Print out sky model in ASCII format. */
    if (type == OSKAR_DOUBLE)
    {
        const double *ra_, *dec_, *I_, *Q_, *U_, *V_, *ref_, *sp_, *rm_;
        const double *maj_, *min_, *pa_;
        ra_  = oskar_mem_double_const(oskar_sky_ra_rad_const(sky), status);
        dec_ = oskar_mem_double_const(oskar_sky_dec_rad_const(sky), status);
        I_   = oskar_mem_double_const(oskar_sky_I_const(sky), status);
        Q_   = oskar_mem_double_const(oskar_sky_Q_const(sky), status);
        U_   = oskar_mem_double_const(oskar_sky_U_const(sky), status);
        V_   = oskar_mem_double_const(oskar_sky_V_const(sky), status);
        ref_ = oskar_mem_double_const(oskar_sky_reference_freq_hz_const(sky), status);
        sp_  = oskar_mem_double_const(oskar_sky_spectral_index_const(sky), status);
        rm_  = oskar_mem_double_const(oskar_sky_rotation_measure_rad_const(sky), status);
        maj_ = oskar_mem_double_const(oskar_sky_fwhm_major_rad_const(sky), status);
        min_ = oskar_mem_double_const(oskar_sky_fwhm_minor_rad_const(sky), status);
        pa_  = oskar_mem_double_const(oskar_sky_position_angle_rad_const(sky), status);

        for (s = 0; s < num_sources; ++s)
        {
            fprintf(file, "% 11.6f,% 11.6f,% 12.6e,% 12.6e,% 12.6e,% 12.6e,"
                    "% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 11.6f\n",
                    ra_[s] * RAD2DEG, dec_[s] * RAD2DEG,
                    I_[s], Q_[s], U_[s], V_[s], ref_[s], sp_[s], rm_[s],
                    maj_[s] * RAD2ARCSEC, min_[s] * RAD2ARCSEC,
                    pa_[s] * RAD2DEG);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *ra_, *dec_, *I_, *Q_, *U_, *V_, *ref_, *sp_, *rm_;
        const float *maj_, *min_, *pa_;
        ra_  = oskar_mem_float_const(oskar_sky_ra_rad_const(sky), status);
        dec_ = oskar_mem_float_const(oskar_sky_dec_rad_const(sky), status);
        I_   = oskar_mem_float_const(oskar_sky_I_const(sky), status);
        Q_   = oskar_mem_float_const(oskar_sky_Q_const(sky), status);
        U_   = oskar_mem_float_const(oskar_sky_U_const(sky), status);
        V_   = oskar_mem_float_const(oskar_sky_V_const(sky), status);
        ref_ = oskar_mem_float_const(oskar_sky_reference_freq_hz_const(sky), status);
        sp_  = oskar_mem_float_const(oskar_sky_spectral_index_const(sky), status);
        rm_  = oskar_mem_float_const(oskar_sky_rotation_measure_rad_const(sky), status);
        maj_ = oskar_mem_float_const(oskar_sky_fwhm_major_rad_const(sky), status);
        min_ = oskar_mem_float_const(oskar_sky_fwhm_minor_rad_const(sky), status);
        pa_  = oskar_mem_float_const(oskar_sky_position_angle_rad_const(sky), status);

        for (s = 0; s < num_sources; ++s)
        {
            fprintf(file, "% 11.6f,% 11.6f,% 12.6e,% 12.6e,% 12.6e,% 12.6e,"
                    "% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 11.6f\n",
                    ra_[s] * RAD2DEG, dec_[s] * RAD2DEG,
                    I_[s], Q_[s], U_[s], V_[s], ref_[s], sp_[s], rm_[s],
                    maj_[s] * RAD2ARCSEC, min_[s] * RAD2ARCSEC,
                    pa_[s] * RAD2DEG);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    fclose(file);
}

#ifdef __cplusplus
}
#endif
