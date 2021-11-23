/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_find_closest_match.h"
#include "math/oskar_random_gaussian.h"
#include "vis/oskar_vis_block.h"
#include "vis/private_vis_block.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_get_station_std_dev_for_channel(oskar_Mem* station_std_dev,
        double frequency_hz, const oskar_Telescope* tel, int* status)
{
    int i = 0, j = 0;
    const oskar_Mem *noise_freq = 0, *noise_rms = 0;

    /* Ensure output array is big enough. */
    const int num_stations = oskar_telescope_num_stations(tel);
    oskar_mem_ensure(station_std_dev, num_stations, status);

    /* Loop over stations and get noise value standard deviation for each. */
    const int* type_map = oskar_mem_int_const(
            oskar_telescope_station_type_map_const(tel), status);
    for (i = 0; i < num_stations; ++i)
    {
        const oskar_Station* station =
                oskar_telescope_station_const(tel, type_map[i]);
        if (!station) station = oskar_telescope_station_const(tel, 0);
        noise_freq = oskar_station_noise_freq_hz_const(station);
        noise_rms = oskar_station_noise_rms_jy_const(station);
        j = oskar_find_closest_match(frequency_hz, noise_freq, status);
        oskar_mem_copy_contents(station_std_dev, noise_rms, i, j, 1, status);
    }
}

/* Applies noise to data in a visibility block, for the given channel. */
static void oskar_vis_block_apply_noise(oskar_VisBlock* vis,
        const oskar_Mem* station_std_dev, unsigned int seed,
        int global_slice_idx, int local_slice_idx,
        double channel_bandwidth_hz, double time_int_sec, int* status)
{
    int a1 = 0, a2 = 0, b = 0, c = 0;
    void *acorr_ptr = 0, *xcorr_ptr = 0;
    double rnd[8];
    const double inv_sqrt2 = 1.0 / sqrt(2.0);

    /* Get pointer to start of block, and block dimensions. */
    acorr_ptr = oskar_mem_void(oskar_vis_block_auto_correlations(vis));
    xcorr_ptr = oskar_mem_void(oskar_vis_block_cross_correlations(vis));
    const int have_autocorr  = oskar_vis_block_has_auto_correlations(vis);
    const int have_crosscorr = oskar_vis_block_has_cross_correlations(vis);
    const int num_baselines  = oskar_vis_block_num_baselines(vis);
    const int num_stations   = oskar_vis_block_num_stations(vis);

    /* Get factor for conversion of sigma to SEFD. */
    const double sefd_factor = sqrt(2.0 * channel_bandwidth_hz * time_int_sec);

    /* If we are adding noise directly to Stokes I, the noise is defined
     * as single dipole noise, so we have to divide by sqrt(2) to take into
     * account of the two different dipoles that go into the calculation of
     * Stokes I. For polarised visibilities this is not required, as this
     * falls out naturally when evaluating Stokes I from the dipole
     * correlations (i.e. I = 0.5 (XX+YY) ). */

    switch (oskar_mem_type(oskar_vis_block_cross_correlations(vis)))
    {
    case OSKAR_SINGLE_COMPLEX:
    {
        const float* st_std = oskar_mem_float_const(station_std_dev, status);
        if (have_crosscorr)
        {
            float2* data = (float2*) xcorr_ptr + (num_baselines * local_slice_idx);
            for (a1 = 0, b = 0; a1 < num_stations; ++a1)
            {
                for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                {
                    oskar_random_gaussian2(seed, c++, global_slice_idx, rnd);
                    const double std = sqrt(st_std[a1] * st_std[a2]) * inv_sqrt2;
                    data[b].x += std * rnd[0];
                    data[b].y += std * rnd[1];
                }
            }
        }

        if (have_autocorr)
        {
            /* Autocorrelation noise. Phases are all zero after
             * autocorrelation, so ignore the imaginary components. */
            float2* data = (float2*) acorr_ptr + (num_stations * local_slice_idx);
            for (a1 = 0; a1 < num_stations; ++a1)
            {
                oskar_random_gaussian2(seed, c++, global_slice_idx, rnd);
                const double std = st_std[a1];
                const double mean = sqrt(2.0) * st_std[a1];
                data[a1].x += std * rnd[0] + mean * sefd_factor;
            }
        }
        break;
    }
    case OSKAR_SINGLE_COMPLEX_MATRIX:
    {
        const float* st_std = oskar_mem_float_const(station_std_dev, status);
        if (have_crosscorr)
        {
            float4c* data = (float4c*) xcorr_ptr + (num_baselines * local_slice_idx);
            for (a1 = 0, b = 0; a1 < num_stations; ++a1)
            {
                for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                {
                    oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd);
                    oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd + 4);
                    const double std = sqrt(st_std[a1] * st_std[a2]);
                    data[b].a.x += std * rnd[0];
                    data[b].a.y += std * rnd[1];
                    data[b].b.x += std * rnd[2];
                    data[b].b.y += std * rnd[3];
                    data[b].c.x += std * rnd[4];
                    data[b].c.y += std * rnd[5];
                    data[b].d.x += std * rnd[6];
                    data[b].d.y += std * rnd[7];
                }
            }
        }

        if (have_autocorr)
        {
            /* Autocorrelation noise. Phases are all zero after
             * autocorrelation, so ignore the imaginary components. */
            float4c* data = (float4c*) acorr_ptr + (num_stations * local_slice_idx);
            for (a1 = 0; a1 < num_stations; ++a1)
            {
                oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd);
                oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd + 4);
                const double std = st_std[a1] * sqrt(2.0);
                const double mean = std * sefd_factor;
                data[a1].a.x += std * rnd[0] + mean;
                data[a1].b.x += std * rnd[1];
                data[a1].b.y += std * rnd[2];
                data[a1].c.x += std * rnd[3];
                data[a1].c.y += std * rnd[4];
                data[a1].d.x += std * rnd[5] + mean;
            }
        }
        break;
    }
    case OSKAR_DOUBLE_COMPLEX:
    {
        const double* st_std = oskar_mem_double_const(station_std_dev, status);
        if (have_crosscorr)
        {
            double2* data = (double2*) xcorr_ptr + (num_baselines * local_slice_idx);
            for (a1 = 0, b = 0; a1 < num_stations; ++a1)
            {
                for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                {
                    oskar_random_gaussian2(seed, c++, global_slice_idx, rnd);
                    const double std = sqrt(st_std[a1] * st_std[a2]) * inv_sqrt2;
                    data[b].x += std * rnd[0];
                    data[b].y += std * rnd[1];
                }
            }
        }

        if (have_autocorr)
        {
            /* Autocorrelation noise. Phases are all zero after
             * autocorrelation, so ignore the imaginary components. */
            double2* data = (double2*) acorr_ptr + (num_stations * local_slice_idx);
            for (a1 = 0; a1 < num_stations; ++a1)
            {
                oskar_random_gaussian2(seed, c++, global_slice_idx, rnd);
                const double std  = st_std[a1];
                const double mean = st_std[a1] * sefd_factor * sqrt(2.0);
                data[a1].x += std * rnd[0] + mean;
            }
        }
        break;
    }
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
    {
        const double* st_std = oskar_mem_double_const(station_std_dev, status);
        if (have_crosscorr)
        {
            double4c* data = (double4c*) xcorr_ptr + (num_baselines * local_slice_idx);
            for (a1 = 0, b = 0; a1 < num_stations; ++a1)
            {
                for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                {
                    oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd);
                    oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd + 4);
                    const double std = sqrt(st_std[a1] * st_std[a2]);
                    data[b].a.x += std * rnd[0];
                    data[b].a.y += std * rnd[1];
                    data[b].b.x += std * rnd[2];
                    data[b].b.y += std * rnd[3];
                    data[b].c.x += std * rnd[4];
                    data[b].c.y += std * rnd[5];
                    data[b].d.x += std * rnd[6];
                    data[b].d.y += std * rnd[7];
                }
            }
        }

        if (have_autocorr)
        {
            /* Autocorrelation noise. Phases are all zero after
             * autocorrelation, so ignore the imaginary components. */
            double4c* data = (double4c*) acorr_ptr + (num_stations * local_slice_idx);
            for (a1 = 0; a1 < num_stations; ++a1)
            {
                oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd);
                oskar_random_gaussian4(seed, c++, global_slice_idx, 0, 0, rnd + 4);
                const double std  = st_std[a1] * sqrt(2.0);
                const double mean = std * sefd_factor;
                data[a1].a.x += std * rnd[0] + mean;
                data[a1].b.x += std * rnd[1];
                data[a1].b.y += std * rnd[2];
                data[a1].c.x += std * rnd[3];
                data[a1].c.y += std * rnd[4];
                data[a1].d.x += std * rnd[5] + mean;
            }
        }
        break;
    }
    };
}

void oskar_vis_block_add_system_noise(oskar_VisBlock* vis,
        const oskar_VisHeader* header, const oskar_Telescope* telescope,
        oskar_Mem* station_work, int* status)
{
    int t = 0, c = 0;
    int num_times_block = 0, num_channels_block = 0, num_channels_total = 0;
    int start_time = 0, start_channel = 0;
    unsigned int seed = 0;
    double freq_start_hz = 0.0, freq_inc_hz = 0.0;
    double channel_bandwidth_hz = 0.0, time_int_sec = 0.0;
    if (*status) return;

    /* Check baseline dimensions match. */
    if (oskar_telescope_num_baselines(telescope) !=
            oskar_vis_block_num_baselines(vis))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get frequency start and increment. */
    seed                 = oskar_telescope_noise_seed(telescope);
    num_times_block      = oskar_vis_block_num_times(vis);
    num_channels_block   = oskar_vis_block_num_channels(vis);
    num_channels_total   = oskar_vis_header_num_channels_total(header);
    start_channel        = oskar_vis_block_start_channel_index(vis);
    start_time           = oskar_vis_block_start_time_index(vis);
    channel_bandwidth_hz = oskar_vis_header_channel_bandwidth_hz(header);
    time_int_sec         = oskar_vis_header_time_average_sec(header);
    freq_start_hz        = oskar_vis_header_freq_start_hz(header);
    freq_inc_hz          = oskar_vis_header_freq_inc_hz(header);

    /* Loop over channels in the block. */
    for (c = 0; c < num_channels_block; ++c)
    {
        const int channel_index = c + start_channel;
        const double freq_hz = freq_start_hz + channel_index * freq_inc_hz;
        oskar_get_station_std_dev_for_channel(station_work, freq_hz,
                telescope, status);

        /* Loop over time samples in the block. */
        for (t = 0; t < num_times_block; ++t)
        {
            /* Get slice indices. */
            const int local_slice_index = t * num_channels_block + c;
            const int global_slice_index =
                    (t + start_time) * num_channels_total + channel_index;

            /* Add noise to the slice. */
            oskar_vis_block_apply_noise(vis, station_work, seed,
                    global_slice_index, local_slice_index,
                    channel_bandwidth_hz, time_int_sec, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
