/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "math/oskar_cmath.h"
#include <stdio.h>

#define RAD2DEG 180.0/M_PI
#define RAD2ARCSEC RAD2DEG * 3600.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_save(const oskar_Sky* sky, const char* filename, int* status)
{
    int i = 0;
    FILE* file = 0;
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
    const int type = oskar_sky_precision(sky);
    const int num_sources = oskar_sky_num_sources(sky);

    /* Print a helpful header. */
    fprintf(file, "# Number of sources: %i\n", num_sources);
    fprintf(file, "# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), "
            "Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2), "
            "FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)\n");

    /* Print out sky model in ASCII format. */
    if (type == OSKAR_DOUBLE)
    {
        const double *ra_ = 0, *dec_ = 0, *I_ = 0, *Q_ = 0, *U_ = 0, *V_ = 0;
        const double *ref_ = 0, *sp_ = 0, *rm_ = 0;
        const double *maj_ = 0, *min_ = 0, *pa_ = 0;
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

        for (i = 0; i < num_sources; ++i)
        {
            fprintf(file, "% 11.6f,% 11.6f,% 12.6e,% 12.6e,% 12.6e,% 12.6e,"
                    "% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 11.6f\n",
                    ra_[i] * RAD2DEG, dec_[i] * RAD2DEG,
                    I_[i], Q_[i], U_[i], V_[i], ref_[i], sp_[i], rm_[i],
                    maj_[i] * RAD2ARCSEC, min_[i] * RAD2ARCSEC,
                    pa_[i] * RAD2DEG);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *ra_ = 0, *dec_ = 0, *I_ = 0, *Q_ = 0, *U_ = 0, *V_ = 0;
        const float *ref_ = 0, *sp_ = 0, *rm_ = 0;
        const float *maj_ = 0, *min_ = 0, *pa_ = 0;
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

        for (i = 0; i < num_sources; ++i)
        {
            fprintf(file, "% 11.6f,% 11.6f,% 12.6e,% 12.6e,% 12.6e,% 12.6e,"
                    "% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 12.6e,% 11.6f\n",
                    ra_[i] * RAD2DEG, dec_[i] * RAD2DEG,
                    I_[i], Q_[i], U_[i], V_[i], ref_[i], sp_[i], rm_[i],
                    maj_[i] * RAD2ARCSEC, min_[i] * RAD2ARCSEC,
                    pa_[i] * RAD2DEG);
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
