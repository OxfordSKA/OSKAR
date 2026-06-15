/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "log/oskar_log.h"
#include "sky/oskar_sky.h"
#include "sky/private_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_check_columns(const oskar_Sky* sky, int* status)
{
    int i = 0, j = 0;
    const char* labels[4] = {"I", "Q", "U", "V"};
    const int num_sources = sky->attr_int[OSKAR_SKY_NUM_SOURCES];
    for (i = 0; i < num_sources; ++i)
    {
        int num_stokes[4] = {0, 0, 0, 0};
        if (*status) break;
        const int num_ra = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_RA_RAD, i
        );
        const int num_dec = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_DEC_RAD, i
        );
        const int num_freq = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_REF_HZ, i
        );
        const int num_alpha = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_SPEC_IDX, i
        );
        const int num_rm = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_RM_RAD, i
        );
        const int num_major = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_MAJOR_RAD, i
        );
        const int num_minor = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_MINOR_RAD, i
        );
        const int num_pa = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_PA_RAD, i
        );
        const int num_lin_si = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_LIN_SI, i
        );
        const int num_pol_ang = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_POLA_RAD, i
        );
        const int num_pol_frac = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_POLF, i
        );
        const int num_ref_wave = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_REF_WAVE_M, i
        );
        const int num_spec_curv = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_SPEC_CURV, i
        );
        const int num_line_width = oskar_sky_num_valid_columns_of_type(
                sky, OSKAR_SKY_LINE_WIDTH_HZ, i
        );
        for (j = 0; j < 4; ++j)
        {
            const oskar_SkyColumn col = (oskar_SkyColumn) (
                    j + (int) OSKAR_SKY_I_JY
            );
            num_stokes[j] = oskar_sky_num_valid_columns_of_type(sky, col, i);
            if ((num_stokes[j] > 1 && num_freq != num_stokes[j]) ||
                    (num_stokes[j] == 1 && num_freq > num_stokes[j]))
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                oskar_log_error(
                        0, "Source %d needs the same number of "
                        "ReferenceFrequency values as Stokes%s values.",
                        i, labels[j]
                );
            }
            if ((num_alpha > 0 || num_spec_curv > 0) && num_stokes[j] > 1)
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                oskar_log_error(
                        0, "Source %d has both a spectral index/curvature term "
                        "and multiple Stokes%s values. Only one can be used.",
                        i, labels[j]
                );
            }
        }
        if (num_ra != 1 || num_dec != 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d needs one RA and one Dec value.", i);
        }
        if (num_stokes[0] == 0)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d needs a StokesI value.", i);
        }
        if (num_freq != 1 && num_alpha > 0)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d needs one ReferenceFrequency "
                    "to use SpectralIndex values.", i
            );
        }
        if (num_rm > 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d can have, at most, one "
                    "RotationMeasure value.", i
            );
        }
        if (num_major > 1 || num_minor > 1 || num_pa > 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d has invalid Gaussian "
                    "MajorAxis/MinorAxis/Orientation values.", i
            );
        }
        if (num_lin_si > 0 && num_alpha < 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d has a LogarithmicSI value "
                    "but no SpectralIndex.", i
            );
        }
        if (num_ref_wave > 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d can have, at most, one "
                    "ReferenceWavelength value.", i
            );
        }
        if (num_spec_curv > 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d can have, at most, one "
                    "SpectralCurvature term.", i
            );
        }
        if (num_spec_curv > 0 && num_alpha != 1)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(0, "Source %d has a SpectralCurvature term, "
                    "so it also requires a single SpectralIndex term.", i
            );
        }
        if (num_pol_ang > 1 && num_pol_ang != num_freq)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d needs the same number of ReferenceFrequency "
                    "values as PolarizationAngle values.", i
            );
        }
        if (num_pol_frac > 1 && num_pol_frac != num_freq)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d needs the same number of ReferenceFrequency "
                    "values as PolarizedFraction values.", i
            );
        }
        if (num_line_width > 1 && num_line_width != num_freq)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d needs the same number of ReferenceFrequency "
                    "values as LineWidth values.", i
            );
        }
        if (num_line_width > 1 && num_line_width != num_stokes[0])
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d needs the same number of StokesI "
                    "values as LineWidth values.", i
            );
        }
        if (num_line_width > 0 && (num_alpha > 0 || num_spec_curv > 0))
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            oskar_log_error(
                    0, "Source %d has both a LineWidth and SpectralIndex "
                    "(or SpectralCurvature) value. Only one can be used.", i
            );
        }
    }
}

#ifdef __cplusplus
}
#endif
