/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <private_sky.h>
#include <oskar_sky.h>

#include <oskar_convert_apparent_ra_dec_to_horizon_direction_cuda.h>
#include <oskar_update_horizon_mask_cuda.h>
#include <oskar_cuda_check_error.h>
#include <oskar_mem.h>

#include <thrust/device_vector.h> // Must be included before thrust/copy.h
#include <thrust/copy.h>

using thrust::copy_if;
using thrust::device_pointer_cast;
using thrust::device_ptr;

struct is_true
{
    __host__ __device__
    bool operator()(const int x) {return (bool)x;}
};

#define DPCT(ptr)  device_pointer_cast((T*) oskar_mem_void(ptr))
#define DPCTC(ptr) device_pointer_cast((const T*) oskar_mem_void_const(ptr))
#define DPT  device_ptr<T>
#define DPTC device_ptr<const T>

template<typename T>
static void copy_source_data(oskar_Sky* output,
        const oskar_Sky* input, const oskar_Mem* mask, int* status)
{
    int num = oskar_sky_num_sources(input);

    // Cast to device pointers.
    DPTC ra_in   = DPCTC(oskar_sky_ra_const(input));
    DPT  ra_out  = DPCT(oskar_sky_ra(output));
    DPTC dec_in  = DPCTC(oskar_sky_dec_const(input));
    DPT  dec_out = DPCT(oskar_sky_dec(output));
    DPTC I_in    = DPCTC(oskar_sky_I_const(input));
    DPT  I_out   = DPCT(oskar_sky_I(output));
    DPTC Q_in    = DPCTC(oskar_sky_Q_const(input));
    DPT  Q_out   = DPCT(oskar_sky_Q(output));
    DPTC U_in    = DPCTC(oskar_sky_U_const(input));
    DPT  U_out   = DPCT(oskar_sky_U(output));
    DPTC V_in    = DPCTC(oskar_sky_V_const(input));
    DPT  V_out   = DPCT(oskar_sky_V(output));
    DPTC ref_in  = DPCTC(oskar_sky_reference_freq_const(input));
    DPT  ref_out = DPCT(oskar_sky_reference_freq(output));
    DPTC sp_in   = DPCTC(oskar_sky_spectral_index_const(input));
    DPT  sp_out  = DPCT(oskar_sky_spectral_index(output));
    DPTC rm_in   = DPCTC(oskar_sky_rotation_measure_const(input));
    DPT  rm_out  = DPCT(oskar_sky_rotation_measure(output));
    DPTC l_in    = DPCTC(oskar_sky_l_const(input));
    DPT  l_out   = DPCT(oskar_sky_l(output));
    DPTC m_in    = DPCTC(oskar_sky_m_const(input));
    DPT  m_out   = DPCT(oskar_sky_m(output));
    DPTC n_in    = DPCTC(oskar_sky_n_const(input));
    DPT  n_out   = DPCT(oskar_sky_n(output));
    DPTC rad_in  = DPCTC(oskar_sky_radius_arcmin_const(input));
    DPT  rad_out = DPCT(oskar_sky_radius_arcmin(output));
    DPTC a_in    = DPCTC(oskar_sky_gaussian_a_const(input));
    DPT  a_out   = DPCT(oskar_sky_gaussian_a(output));
    DPTC b_in    = DPCTC(oskar_sky_gaussian_b_const(input));
    DPT  b_out   = DPCT(oskar_sky_gaussian_b(output));
    DPTC c_in    = DPCTC(oskar_sky_gaussian_c_const(input));
    DPT  c_out   = DPCT(oskar_sky_gaussian_c(output));
    DPTC maj_in  = DPCTC(oskar_sky_fwhm_major_const(input));
    DPT  maj_out = DPCT(oskar_sky_fwhm_major(output));
    DPTC min_in  = DPCTC(oskar_sky_fwhm_minor_const(input));
    DPT  min_out = DPCT(oskar_sky_fwhm_minor(output));
    DPTC pa_in   = DPCTC(oskar_sky_position_angle_const(input));
    DPT  pa_out  = DPCT(oskar_sky_position_angle(output));
    device_ptr<const int> m = device_pointer_cast(
            oskar_mem_int_const(mask, status));

    // Copy sources to new model based on mask values.
    DPT out = copy_if(ra_in, ra_in + num, m, ra_out, is_true());
    copy_if(dec_in, dec_in + num, m, dec_out, is_true());
    copy_if(I_in, I_in + num, m, I_out, is_true());
    copy_if(Q_in, Q_in + num, m, Q_out, is_true());
    copy_if(U_in, U_in + num, m, U_out, is_true());
    copy_if(V_in, V_in + num, m, V_out, is_true());
    copy_if(ref_in, ref_in + num, m, ref_out, is_true());
    copy_if(sp_in, sp_in + num, m, sp_out, is_true());
    copy_if(rm_in, rm_in + num, m, rm_out, is_true());
    copy_if(l_in, l_in + num, m, l_out, is_true());
    copy_if(m_in, m_in + num, m, m_out, is_true());
    copy_if(n_in, n_in + num, m, n_out, is_true());
    copy_if(rad_in, rad_in + num, m, rad_out, is_true());
    copy_if(a_in, a_in + num, m, a_out, is_true());
    copy_if(b_in, b_in + num, m, b_out, is_true());
    copy_if(c_in, c_in + num, m, c_out, is_true());
    copy_if(maj_in, maj_in + num, m, maj_out, is_true());
    copy_if(min_in, min_in + num, m, min_out, is_true());
    copy_if(pa_in, pa_in + num, m, pa_out, is_true());

    // Get the number of sources above the horizon.
    // Don't call resize, since that is too expensive.
    output->num_sources = out - ra_out;
}

extern "C"
void oskar_sky_horizon_clip(oskar_Sky* output,
        const oskar_Sky* input, const oskar_Telescope* telescope,
        double gast, oskar_StationWork* work, int* status)
{
    int *mask, type;
    const oskar_Station* s;
    oskar_Mem *horizon_mask, *hor_x, *hor_y, *hor_z;

    /* Check all inputs. */
    if (!output || !input || !telescope || !work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get pointers to work arrays. */
    horizon_mask = oskar_station_work_horizon_mask(work);
    hor_x = oskar_station_work_source_horizontal_x(work);
    hor_y = oskar_station_work_source_horizontal_y(work);
    hor_z = oskar_station_work_source_horizontal_z(work);

    /* Check that the types match. */
    type = oskar_sky_type(input);
    if (oskar_sky_type(output) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check for the correct location. */
    if (oskar_sky_location(output) != OSKAR_LOCATION_GPU ||
            oskar_sky_location(input) != OSKAR_LOCATION_GPU ||
            oskar_mem_location(hor_x) != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Copy extended source flag. */
    oskar_sky_set_use_extended(output,
            oskar_sky_use_extended(input));

    /* Get the data dimensions. */
    int num_sources = oskar_sky_num_sources(input);
    int num_stations = oskar_telescope_num_stations(telescope);

    /* Resize the output structure if necessary. */
    if (oskar_sky_num_sources(output) < num_sources)
        oskar_sky_resize(output, num_sources, status);

    /* Resize the work buffers if necessary. */
    if ((int)oskar_mem_length(horizon_mask) < num_sources)
        oskar_mem_realloc(horizon_mask, num_sources, status);
    if ((int)oskar_mem_length(hor_x) < num_sources)
        oskar_mem_realloc(hor_x, num_sources, status);
    if ((int)oskar_mem_length(hor_y) < num_sources)
        oskar_mem_realloc(hor_y, num_sources, status);
    if ((int)oskar_mem_length(hor_z) < num_sources)
        oskar_mem_realloc(hor_z, num_sources, status);

    /* Clear horizon mask. */
    oskar_mem_clear_contents(horizon_mask, status);
    mask = oskar_mem_int(horizon_mask, status);

    /* Check if safe to proceed. */
    if (*status) return;

    if (type == OSKAR_SINGLE)
    {
        const float *ra, *dec;
        float *x, *y, *z;
        ra   = oskar_mem_float_const(oskar_sky_ra_const(input), status);
        dec  = oskar_mem_float_const(oskar_sky_dec_const(input), status);
        x    = oskar_mem_float(hor_x, status);
        y    = oskar_mem_float(hor_y, status);
        z    = oskar_mem_float(hor_z, status);

        /* Create the mask. */
        for (int i = 0; i < num_stations; ++i)
        {
            /* Get the station position. */
            double longitude, latitude, lst;
            s = oskar_telescope_station_const(telescope, i);
            longitude = oskar_station_longitude_rad(s);
            latitude  = oskar_station_latitude_rad(s);
            lst = gast + longitude;

            /* Evaluate source horizontal x,y,z direction cosines. */
            oskar_convert_apparent_ra_dec_to_horizon_direction_cuda_f(
                    num_sources, ra, dec, lst, latitude, x, y, z);

            /* Update the mask. */
            oskar_update_horizon_mask_cuda_f(num_sources, mask,
                    (const float*)z);
        }

        /* Copy out source data based on the mask values. */
        copy_source_data<float>(output, input, horizon_mask, status);
        oskar_cuda_check_error(status);
    }
    else if (type == OSKAR_DOUBLE)
    {
        const double *ra, *dec;
        double *x, *y, *z;
        ra   = oskar_mem_double_const(oskar_sky_ra_const(input), status);
        dec  = oskar_mem_double_const(oskar_sky_dec_const(input), status);
        x    = oskar_mem_double(hor_x, status);
        y    = oskar_mem_double(hor_y, status);
        z    = oskar_mem_double(hor_z, status);

        /* Create the mask. */
        for (int i = 0; i < num_stations; ++i)
        {
            /* Get the station position. */
            double longitude, latitude, lst;
            s = oskar_telescope_station_const(telescope, i);
            longitude = oskar_station_longitude_rad(s);
            latitude  = oskar_station_latitude_rad(s);
            lst = gast + longitude;

            /* Evaluate source horizontal x,y,z direction cosines. */
            oskar_convert_apparent_ra_dec_to_horizon_direction_cuda_d(
                    num_sources, ra, dec, lst, latitude, x, y, z);

            /* Update the mask. */
            oskar_update_horizon_mask_cuda_d(num_sources, mask,
                    (const double*)z);
        }

        /* Copy out source data based on the mask values. */
        copy_source_data<double>(output, input, horizon_mask, status);
        oskar_cuda_check_error(status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}
