/*
 * Copyright (c) 2016, The University of Oxford
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

#include <fitsio.h>
#include <cufft.h>
#include <oskar_mem.h>
#include <oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per GPU. */
struct DeviceData
{
    oskar_Mem *l, *m, *n;
    oskar_Mem *uu, *vv, *ww, *amp;
    oskar_Mem *block_gpu, *block_cpu, *plane_gpu;
};
typedef struct DeviceData DeviceData;

struct oskar_Imager
{
    fitsfile* fits_file[4];
    oskar_Log* log;

    /* Settings parameters. */
    int imager_prec, num_gpus, *cuda_device_ids;
    int chan_snaps, time_snaps, chan_range[2], time_range[2];
    int im_type, im_num_times, im_num_channels, im_num_pols, pol_offset;
    int algorithm, size, num_pixels, use_ms, use_stokes, support, oversample;
    char direction_type, kernel_type, *image_root, *ms_column;
    double fov_deg, im_centre_deg[2];

    /* Visibility meta-data. */
    int vis_time_range[2], vis_chan_range[2], num_stations;
    double vis_centre_deg[2];
    double vis_freq_start_hz, im_freq_start_hz, freq_inc_hz;
    double vis_time_start_mjd_utc, im_time_start_mjd_utc, time_inc_sec;
    oskar_Mem *st_x, *st_y, *st_z; /* Station coordinates. */

    /* Scratch data. */
    oskar_Mem *uu_im, *vv_im, *ww_im, *vis_im;
    oskar_Mem *uu_tmp, *vv_tmp, *ww_tmp, *stokes;
    oskar_Mem *uu_rot, *vv_rot, *ww_rot, *work_uvw;
    int num_planes; /* for each time, channel and polarisation. */
    double* plane_norm, delta_l, delta_m, delta_n;
    oskar_Mem **planes, *plane_tmp;

    /* DFT imager data. */
    oskar_Mem *l, *m, *n;

    /* FFT imager data. */
    double cellsize_rad;
    oskar_Mem *conv_func, *corr_func;
    cufftHandle cufft_plan_imager;

    /* W-projection imager data. */
    int num_w_kernels, *w_support;
    double w_min, w_max;
    oskar_Mem **w_kernels;

    /* Memory allocated per GPU (array of DeviceData structures). */
    DeviceData* d;
};
#ifndef OSKAR_IMAGER_TYPEDEF_
#define OSKAR_IMAGER_TYPEDEF_
typedef struct oskar_Imager oskar_Imager;
#endif /* OSKAR_IMAGER_TYPEDEF_ */

#ifdef __cplusplus
}
#endif
