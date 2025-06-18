/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#include "telescope/station/element/oskar_apply_element_taper_cosine.h"
#include "telescope/station/element/oskar_apply_element_taper_gaussian.h"
#include "telescope/station/element/oskar_evaluate_dipole_pattern.h"
#include "telescope/station/element/oskar_evaluate_spherical_wave_sum.h"
#include "telescope/station/element/oskar_evaluate_spherical_wave_sum_feko.h"
#include "telescope/station/element/oskar_evaluate_spherical_wave_sum_galileo.h"
#include "convert/oskar_convert_enu_directions_to_theta_phi.h"
#include "math/oskar_find_closest_match.h"

#include "math/oskar_cmath.h"

#define C_0 299792458.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_evaluate(
        const oskar_Element* model,
        int normalise,
        int swap_xy,
        double orientation_x,
        double orientation_y,
        int offset_points,
        int num_points,
        const oskar_Mem* x,
        const oskar_Mem* y,
        const oskar_Mem* z,
        double frequency_hz,
        oskar_Mem* theta,
        oskar_Mem* phi_x,
        oskar_Mem* phi_y,
        int offset_out,
        oskar_Mem* output,
        int* status)
{
    double dipole_length_m = 0.0;
    if (*status) return;
    if (!oskar_mem_is_complex(output))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    const int num_points_norm = normalise ? 1 + num_points : num_points;
    oskar_mem_ensure(output, num_points_norm + offset_out, status);
    oskar_mem_ensure(theta, num_points_norm, status);
    oskar_mem_ensure(phi_x, num_points_norm, status);
    oskar_mem_ensure(phi_y, num_points_norm, status);

    /* Get the element model properties. */
    const int element_type = model->element_type;
    const int taper_type   = model->taper_type;
    dipole_length_m = model->dipole_length;
    if (model->dipole_length_units == OSKAR_WAVELENGTHS)
    {
        dipole_length_m *= (C_0 / frequency_hz);
    }

    /* Compute (effective) theta and phi coordinates. */
    /*
     * TODO CHECK: delta_phi values may need to be negative to match proposed
     * changes in orientation angle in
     * oskar_evaluate_station_beam_aperture_array().
     */
    oskar_convert_enu_directions_to_theta_phi(
            offset_points, num_points, x, y, z, normalise,
            (M_PI/2) - orientation_x,
            (M_PI/2) - orientation_y,
            theta, phi_x, phi_y, status
    );

    /* Check if element type is isotropic. */
    if (element_type == OSKAR_ELEMENT_TYPE_ISOTROPIC)
    {
        oskar_mem_set_value_real(
                output, 1.0, offset_out, num_points_norm, status
        );
    }
    else if (oskar_mem_is_matrix(output))
    {
        /* Evaluate polarised response if output array is matrix type. */
        const int id = oskar_find_closest_match_d(
                frequency_hz,
                oskar_element_num_freq(model),
                oskar_element_freqs_hz_const(model)
        );
        if (oskar_element_has_spherical_wave_data(model, id))
        {
            oskar_evaluate_spherical_wave_sum(
                    num_points_norm, theta, phi_x,
                    (model->common_phi_coords[id] ? phi_x : phi_y),
                    model->l_max[id], model->sph_wave[id],
                    swap_xy, offset_out, output, status
            );
        }
        else if (oskar_element_has_spherical_wave_feko_data(model, id))
        {
            oskar_evaluate_spherical_wave_sum_feko(
                    num_points_norm, theta, phi_x,
                    (model->common_phi_coords[id] ? phi_x : phi_y),
                    model->l_max[id], model->sph_wave_feko[id],
                    swap_xy, offset_out, output, status
            );
        }
        else if (oskar_element_has_spherical_wave_galileo_data(model, id))
        {
            oskar_evaluate_spherical_wave_sum_galileo(
                    num_points_norm, theta, phi_x,
                    (model->common_phi_coords[id] ? phi_x : phi_y),
                    model->l_max[id], model->sph_wave_galileo[id],
                    swap_xy, offset_out, output, status
            );
        }
        else if (element_type == OSKAR_ELEMENT_TYPE_DIPOLE)
        {
            oskar_evaluate_dipole_pattern(
                    num_points_norm, theta, phi_x, phi_y,
                    frequency_hz, dipole_length_m,
                    swap_xy, offset_out, output, status
            );
        }
    }
    else /* Scalar response. */
    {
        if (element_type == OSKAR_ELEMENT_TYPE_DIPOLE)
        {
            oskar_evaluate_dipole_pattern(
                    num_points_norm, theta, phi_x, phi_y, frequency_hz,
                    dipole_length_m, swap_xy, offset_out, output, status
            );
        }
    }

    /* Apply element pattern normalisation, if specified. */
    if (normalise)
    {
        oskar_mem_normalise(
                output, offset_out, num_points, offset_out + num_points, status
        );
    }

    /* Apply element tapering, if specified. */
    if (taper_type == OSKAR_ELEMENT_TAPER_COSINE)
    {
        oskar_apply_element_taper_cosine(
                num_points, model->cosine_power, theta,
                offset_out, output, status
        );
    }
    else if (taper_type == OSKAR_ELEMENT_TAPER_GAUSSIAN)
    {
        oskar_apply_element_taper_gaussian(
                num_points, model->gaussian_fwhm_rad, theta,
                offset_out, output, status
        );
    }
}

#ifdef __cplusplus
}
#endif
