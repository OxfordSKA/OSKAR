/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_imager.h"

#include <cstdlib>
#include <cstring>

using namespace std;

oskar_Imager* oskar_settings_to_imager(oskar::SettingsTree* s,
        oskar_Log* log, int* status)
{
    (void) log;
    if (*status || !s) return 0;
    s->clear_group();

    // Create and set up the imager.
    oskar_Imager* h = oskar_imager_create(
            s->to_int("image/double_precision", status) ?
            OSKAR_DOUBLE : OSKAR_SINGLE, status);
    oskar_log_set_keep_file(oskar_imager_log(h), 0);

    // Set GPU IDs.
    s->begin_group("image");
    if (!s->to_int("use_gpus", status))
    {
        oskar_imager_set_gpus(h, 0, 0, status);
    }
    else
    {
        if (s->starts_with("cuda_device_ids", "all", status))
        {
            oskar_imager_set_gpus(h, -1, 0, status);
        }
        else
        {
            int size = 0;
            const int* ids = s->to_int_list("cuda_device_ids", &size, status);
            if (size > 0)
            {
                oskar_imager_set_gpus(h, size, ids, status);
            }
        }
    }
    if (s->starts_with("num_devices", "auto", status))
    {
        oskar_imager_set_num_devices(h, -1);
    }
    else
    {
        oskar_imager_set_num_devices(h, s->to_int("num_devices", status));
    }

    // Set input and output files.
    int num_files = 0;
    const char* const* files =
            s->to_string_list("input_vis_data", &num_files, status);
    oskar_imager_set_input_files(h, num_files, files, status);
    oskar_imager_set_scale_norm_with_num_input_files(h,
            s->to_int("scale_norm_with_num_input_files", status));
    oskar_imager_set_ms_column(h,
            s->to_string("ms_column", status), status);
    oskar_imager_set_output_root(h, s->to_string("root_path", status));

    // Set remaining imager options.
    oskar_imager_set_image_type(h,
            s->to_string("image_type", status), status);
    if (s->to_int("specify_cellsize", status))
    {
        oskar_imager_set_cellsize(h, s->to_double("cellsize_arcsec", status));
    }
    else
    {
        oskar_imager_set_fov(h, s->to_double("fov_deg", status));
    }
    oskar_imager_set_size(h, s->to_int("size", status), status);
    oskar_imager_set_channel_snapshots(h,
            s->to_int("channel_snapshots", status));
    oskar_imager_set_freq_min_hz(h, s->to_double("freq_min_hz", status));
    oskar_imager_set_freq_max_hz(h, s->to_double("freq_max_hz", status));
    oskar_imager_set_time_min_utc(h, s->to_double("time_min_utc", status));
    oskar_imager_set_time_max_utc(h, s->to_double("time_max_utc", status));
    oskar_imager_set_uv_filter_min(h, s->to_double("uv_filter_min", status));
    oskar_imager_set_uv_filter_max(h, s->to_double("uv_filter_max", status));
    oskar_imager_set_algorithm(h,
            s->to_string("algorithm", status), status);
    oskar_imager_set_weighting(h,
            s->to_string("weighting", status), status);
    oskar_imager_set_uv_taper(h,
            s->to_double("weight_taper/u_wavelengths", status),
            s->to_double("weight_taper/v_wavelengths", status));
    if (s->starts_with("algorithm", "FFT", status) ||
            s->starts_with("algorithm", "fft", status))
    {
        oskar_imager_set_grid_kernel(h,
                s->to_string("fft/kernel_type", status),
                s->to_int("fft/support", status),
                s->to_int("fft/oversample", status), status);
    }
    if (!s->starts_with("wproj/num_w_planes", "auto", status))
    {
        oskar_imager_set_num_w_planes(h,
                s->to_int("wproj/num_w_planes", status));
    }
    oskar_imager_set_fft_on_gpu(h, s->to_int("fft/use_gpu", status));
    oskar_imager_set_grid_on_gpu(h, s->to_int("fft/grid_on_gpu", status));
    oskar_imager_set_generate_w_kernels_on_gpu(h,
            s->to_int("wproj/generate_w_kernels_on_gpu", status));
    if (s->first_letter("direction", status) == 'R')
    {
        oskar_imager_set_direction(h,
                s->to_double("direction/ra_deg", status),
                s->to_double("direction/dec_deg", status));
    }

    // Return handle to imager.
    s->clear_group();
    return h;
}
