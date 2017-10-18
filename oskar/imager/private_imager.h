/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
#include <cufft.h>
#endif

#include <fitsio.h>
#include <mem/oskar_mem.h>
#include <log/oskar_log.h>
#include <utility/oskar_thread.h>
#include <utility/oskar_timer.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per GPU. */
struct DeviceData
{
    oskar_Mem *uu, *vv, *ww, *amp, *weight, *l, *m, *n;
    oskar_Mem *block_dev, *block_cpu;
};
typedef struct DeviceData DeviceData;

struct oskar_Imager
{
    char* output_name[4];
    fitsfile* fits_file[4];
    oskar_Log* log;
    oskar_Timer *tmr_grid_update, *tmr_grid_finalise, *tmr_init;
    oskar_Timer *tmr_read, *tmr_write;

    /* Settings parameters. */
    int imager_prec, num_devices, num_gpus, *gpu_ids, fft_on_gpu;
    int chan_snaps, im_type, num_im_channels, num_im_pols, pol_offset;
    int algorithm, image_size, use_stokes, support, oversample;
    int generate_w_kernels_on_gpu, set_cellsize, set_fov, weighting;
    int num_files, scale_norm_with_num_input_files;
    char direction_type, kernel_type;
    char **input_files, *input_root, *output_root, *ms_column;
    double cellsize_rad, fov_deg, image_padding, im_centre_deg[2];
    double uv_filter_min, uv_filter_max;
    double time_min_utc, time_max_utc, freq_min_hz, freq_max_hz;

    /* Visibility meta-data. */
    int num_sel_freqs;
    double *im_freqs, *sel_freqs;
    double vis_freq_start_hz, freq_inc_hz;

    /* State. */
    int status, i_block;
    oskar_Mutex* mutex;

    /* Scratch data. */
    oskar_Mem *uu_im, *vv_im, *ww_im, *vis_im, *weight_im, *time_im;
    oskar_Mem *uu_tmp, *vv_tmp, *ww_tmp, *stokes, *weight_tmp;
    int coords_only; /* Set if doing a first pass for uniform weighting. */
    int num_planes; /* For each output channel and polarisation. */
    double *plane_norm, delta_l, delta_m, delta_n, M[9];
    oskar_Mem **planes, **weights_grids;

    /* DFT imager data. */
    oskar_Mem *l, *m, *n;

    /* FFT imager data. */
    int grid_size;
    oskar_Mem *conv_func, *corr_func, *fftpack_wsave, *fftpack_work;
#ifdef OSKAR_HAVE_CUDA
    cufftHandle cufft_plan;
#endif

    /* W-projection imager data. */
    size_t ww_points;
    int num_w_planes, conv_size_half;
    double w_scale, ww_min, ww_max, ww_rms;
    oskar_Mem *w_kernels, *w_support;

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
