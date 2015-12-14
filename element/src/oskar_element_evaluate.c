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

#include <private_element.h>
#include <oskar_element.h>

#include <oskar_apply_element_taper_cosine.h>
#include <oskar_apply_element_taper_gaussian.h>
#include <oskar_evaluate_dipole_pattern.h>
#include <oskar_evaluate_geometric_dipole_pattern.h>
#include <oskar_convert_enu_directions_to_theta_phi.h>
#include <oskar_convert_ludwig3_to_theta_phi_components.h>
#include <oskar_find_closest_match.h>

#include <oskar_cmath.h>

#define C_0 299792458.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_evaluate(const oskar_Element* model, oskar_Mem* output,
        double orientation_x, double orientation_y, int num_points,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double frequency_hz, oskar_Mem* theta, oskar_Mem* phi, int* status)
{
    int element_type, taper_type, freq_id;
    double dipole_length_m;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the output array is complex. */
    if (!oskar_mem_is_complex(output))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Resize output array if required. */
    if ((int)oskar_mem_length(output) < num_points)
        oskar_mem_realloc(output, num_points, status);

    /* Get the element model properties. */
    element_type    = model->element_type;
    taper_type      = model->taper_type;
    dipole_length_m = model->dipole_length;
    if (model->dipole_length_units == OSKAR_WAVELENGTHS)
        dipole_length_m *= (C_0 / frequency_hz);

    /* Check if element type is isotropic. */
    if (element_type == OSKAR_ELEMENT_TYPE_ISOTROPIC)
        oskar_mem_set_value_real(output, 1.0, 0, 0, status);

    /* Ensure there is enough space in the theta and phi work arrays. */
    if ((int)oskar_mem_length(theta) < num_points)
        oskar_mem_realloc(theta, num_points, status);
    if ((int)oskar_mem_length(phi) < num_points)
        oskar_mem_realloc(phi, num_points, status);

    /* Compute modified theta and phi coordinates for dipole X. */
    oskar_convert_enu_directions_to_theta_phi(num_points, x, y, z,
            M_PI_2 - orientation_x, theta, phi, status);

    /* Evaluate polarised response if output array is matrix type. */
    if (oskar_mem_is_matrix(output))
    {
        /* Check if spline data present for dipole X. */
        if (oskar_element_has_x_spline_data(model))
        {
            /* Get the frequency index. */
            freq_id = oskar_find_closest_match_d(frequency_hz,
                    oskar_element_num_freq(model),
                    oskar_element_freqs_hz_const(model));

            /* Evaluate spline pattern for dipole X. */
            oskar_splines_evaluate(output, 0, 8, model->x_h_re[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 1, 8, model->x_h_im[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 2, 8, model->x_v_re[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 3, 8, model->x_v_im[freq_id],
                    num_points, theta, phi, status);

            /* Convert from Ludwig-3 to spherical representation. */
            oskar_convert_ludwig3_to_theta_phi_components(output, 0, 4,
                    num_points, phi, status);
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_DIPOLE)
        {
            /* Evaluate dipole pattern for dipole X. */
            oskar_evaluate_dipole_pattern(output, num_points, theta, phi,
                    frequency_hz, dipole_length_m, 0, 4, status);
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE)
        {
            /* Evaluate dipole pattern for dipole X. */
            oskar_evaluate_geometric_dipole_pattern(output, num_points,
                    theta, phi, 0, 4, status);
        }

        /* Compute modified theta and phi coordinates for dipole Y. */
        oskar_convert_enu_directions_to_theta_phi(num_points, x, y, z,
                M_PI_2 - orientation_y, theta, phi, status);

        /* Check if spline data present for dipole Y. */
        if (oskar_element_has_y_spline_data(model))
        {
            /* Get the frequency index. */
            freq_id = oskar_find_closest_match_d(frequency_hz,
                    oskar_element_num_freq(model),
                    oskar_element_freqs_hz_const(model));

            /* Evaluate spline pattern for dipole Y. */
            oskar_splines_evaluate(output, 4, 8, model->y_h_re[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 5, 8, model->y_h_im[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 6, 8, model->y_v_re[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 7, 8, model->y_v_im[freq_id],
                    num_points, theta, phi, status);

            /* Convert from Ludwig-3 to spherical representation. */
            oskar_convert_ludwig3_to_theta_phi_components(output, 2, 4,
                    num_points, phi, status);
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_DIPOLE)
        {
            /* Evaluate dipole pattern for dipole Y. */
            oskar_evaluate_dipole_pattern(output, num_points, theta, phi,
                    frequency_hz, dipole_length_m, 2, 4, status);
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE)
        {
            /* Evaluate dipole pattern for dipole Y. */
            oskar_evaluate_geometric_dipole_pattern(output, num_points,
                    theta, phi, 2, 4, status);
        }
    }

    /* Scalar response. */
    else
    {
        /* Check if scalar spline data present. */
        if (oskar_element_has_scalar_spline_data(model))
        {
            /* Get the frequency index. */
            freq_id = oskar_find_closest_match_d(frequency_hz,
                    oskar_element_num_freq(model),
                    oskar_element_freqs_hz_const(model));

            oskar_splines_evaluate(output, 0, 2, model->scalar_re[freq_id],
                    num_points, theta, phi, status);
            oskar_splines_evaluate(output, 1, 2, model->scalar_im[freq_id],
                    num_points, theta, phi, status);
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_DIPOLE)
        {
            oskar_evaluate_dipole_pattern(output, num_points, theta, phi,
                    frequency_hz, dipole_length_m, 0, 1, status);
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE)
        {
            oskar_evaluate_geometric_dipole_pattern(output, num_points,
                    theta, phi, 0, 1, status);
        }
    }

    /* Apply element tapering, if specified. */
    if (taper_type == OSKAR_ELEMENT_TAPER_COSINE)
    {
        oskar_apply_element_taper_cosine(output, num_points,
                model->cosine_power, theta, status);
    }
    else if (taper_type == OSKAR_ELEMENT_TAPER_GAUSSIAN)
    {
        oskar_apply_element_taper_gaussian(output, num_points,
                model->gaussian_fwhm_rad, theta, status);
    }
}

#ifdef __cplusplus
}
#endif
