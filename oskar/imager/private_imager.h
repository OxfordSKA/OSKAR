/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <fitsio.h>
#include <log/oskar_log.h>
#include <math/oskar_fft.h>
#include <mem/oskar_mem.h>
#include <utility/oskar_thread.h>
#include <utility/oskar_timer.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory allocated per GPU. */
struct DeviceData
{
    /* Scratch data. */
    oskar_Mem *uu, *vv, *ww, *vis, *weight, *counter, *count_skipped;
    oskar_Mem *norm, *num_points_in_tiles, *tile_offsets, *tile_locks;
    oskar_Mem *sorted_uu, *sorted_vv, *sorted_ww;
    oskar_Mem *sorted_wt, *sorted_vis, *sorted_tile;
    int num_planes;
    oskar_Mem **planes;

    /* FFT imager data. */
    oskar_Mem *conv_func;

    /* W-projection imager data. */
    oskar_Mem *w_support, *w_kernels_compact, *w_kernel_start;
};
typedef struct DeviceData DeviceData;

struct oskar_Imager
{
    char* output_name[4];
    fitsfile* fits_file[4];
    oskar_Timer *tmr_overall, *tmr_grid_update, *tmr_grid_finalise, *tmr_init;
    oskar_Timer *tmr_select_scale, *tmr_filter, *tmr_read, *tmr_write;
    oskar_Timer *tmr_copy_convert, *tmr_coord_scan, *tmr_rotate;
    oskar_Timer *tmr_weights_grid, *tmr_weights_lookup;

    /* Settings parameters. */
    int imager_prec, num_devices, num_gpus_avail, dev_loc, num_gpus, *gpu_ids;
    int chan_snaps, im_type, num_im_channels, num_im_pols, pol_offset;
    int algorithm, fft_on_gpu, grid_on_gpu;
    int image_size, use_stokes, support, oversample;
    int generate_w_kernels_on_gpu, set_cellsize, set_fov, weighting;
    int num_files, scale_norm_with_num_input_files;
    char direction_type, kernel_type;
    char **input_files, *input_root, *output_root, *ms_column;
    double cellsize_rad, fov_deg, image_padding, im_centre_deg[2];
    double uv_filter_min, uv_filter_max, uv_taper[2];
    double time_min_utc, time_max_utc, freq_min_hz, freq_max_hz;

    /* Visibility meta-data. */
    int num_sel_freqs;
    double *im_freqs, *sel_freqs;
    double vis_freq_start_hz, freq_inc_hz;

    /* State. */
    int init, status, i_block;
    int coords_only; /* Set if doing a first pass for uniform weighting. */
    oskar_Mutex* mutex;
    oskar_Log* log;
    size_t num_vis_processed;

    /* Scratch data. */
    oskar_Mem *uu_im, *vv_im, *ww_im, *vis_im, *weight_im, *time_im;
    oskar_Mem *uu_tmp, *vv_tmp, *ww_tmp, *stokes, *weight_tmp;
    int num_planes; /* For each output channel and polarisation. */
    double *plane_norm, delta_l, delta_m, delta_n, M[9];
    oskar_Mem **planes, **weights_grids, **weights_guard;

    /* DFT imager data. */
    oskar_Mem *l, *m, *n;

    /* FFT imager data. */
    oskar_FFT* fft;
    int grid_size;
    oskar_Mem *conv_func, *corr_func;

    /* W-projection imager data. */
    size_t ww_points;
    int num_w_planes;
    double w_scale, ww_min, ww_max, ww_rms;
    oskar_Mem *w_support, *w_kernels_compact, *w_kernel_start;

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
