/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"

#include "imager/oskar_grid_weights.h"
#include "imager/oskar_imager.h"
#include "imager/private_imager_create_fits_files.h"
#include "imager/private_imager_filter_time.h"
#include "imager/private_imager_filter_uv.h"
#include "imager/private_imager_select_data.h"
#include "imager/private_imager_set_num_planes.h"
#include "imager/private_imager_taper_weights.h"
#include "imager/private_imager_update_plane_dft.h"
#include "imager/private_imager_update_plane_fft.h"
#include "imager/private_imager_update_plane_wproj.h"
#include "imager/private_imager_weight_radial.h"
#include "imager/private_imager_weight_uniform.h"
#include "log/oskar_log.h"
#include "utility/oskar_device.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_imager_allocate_planes(oskar_Imager* h, int *status);
static void oskar_imager_update_weights_grid(oskar_Imager* h,
        size_t num_points, const oskar_Mem* uu, const oskar_Mem* vv,
        const oskar_Mem* ww, const oskar_Mem* weight, oskar_Mem* weights_grid,
        oskar_Mem* weights_guard, int* status);

void oskar_imager_update_from_block(oskar_Imager* h,
        const oskar_VisHeader* hdr, oskar_VisBlock* block,
        int* status)
{
    int c = 0, t = 0;
    double time_start_mjd = 0.0, time_inc_sec = 0.0;
    oskar_Mem *weight = 0, *weight_ptr = 0, *time_centroid = 0;
    if (*status) return;

    /* Check that cross-correlations exist. */
    if (!oskar_vis_block_has_cross_correlations(block))
    {
        return;
    }

    /* Get dimensions from the block. */
    const int start_time    = oskar_vis_block_start_time_index(block);
    const int start_chan    = oskar_vis_block_start_channel_index(block);
    const int num_baselines = oskar_vis_block_num_baselines(block);
    const int num_channels  = oskar_vis_block_num_channels(block);
    const int num_pols      = oskar_vis_block_num_pols(block);
    const int num_times     = oskar_vis_block_num_times(block);
    const size_t num_rows   = num_baselines * num_times;

    /* Get visibility meta-data. */
    const double freq_inc_hz = oskar_vis_header_freq_inc_hz(hdr);
    const double freq_start_hz = oskar_vis_header_freq_start_hz(hdr);
    time_start_mjd = oskar_vis_header_time_start_mjd_utc(hdr) * 86400.0;
    time_inc_sec = oskar_vis_header_time_inc_sec(hdr);
    oskar_imager_set_vis_frequency(h, freq_start_hz, freq_inc_hz,
            oskar_vis_header_num_channels_total(hdr));
    oskar_imager_set_vis_phase_centre(h,
            oskar_vis_header_phase_centre_ra_deg(hdr),
            oskar_vis_header_phase_centre_dec_deg(hdr));

    /* Weights are all 1. */
    if (!weight)
    {
        size_t num_weights = num_rows * num_pols;
        weight = oskar_mem_create(oskar_mem_precision(
                oskar_vis_block_cross_correlations_const(block)),
                OSKAR_CPU, num_weights, status);
        oskar_mem_set_value_real(weight, 1.0, 0, num_weights, status);
        weight_ptr = weight;
    }

    /* Fill in the time centroid values. */
    time_centroid = oskar_mem_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_rows, status);
    for (t = 0; t < num_times; ++t)
    {
        oskar_mem_set_value_real(time_centroid,
                time_start_mjd + (start_time + t + 0.5) * time_inc_sec,
                t * num_baselines, num_baselines, status);
    }

    /* Get baseline coordinates if required. */
    if (oskar_vis_block_has_station_coords(block))
    {
        oskar_vis_block_station_to_baseline_coords(block, status);
    }

    /* Update the imager with the data. */
    if (!h->coords_only)
    {
        oskar_Mem* scratch = oskar_mem_create(oskar_mem_type(
                oskar_vis_block_cross_correlations_const(block)),
                OSKAR_CPU, num_rows, status);
        for (c = 0; c < num_channels; ++c)
        {
            /* Update per channel. */
            const double freq_hz =
                    freq_start_hz + (start_chan + c) * freq_inc_hz;
            if (freq_hz >= h->freq_min_hz &&
                    (freq_hz <= h->freq_max_hz || h->freq_max_hz == 0.0))
            {
                oskar_timer_resume(h->tmr_copy_convert);
                for (t = 0; t < num_times; ++t)
                {
                    oskar_mem_copy_contents(scratch,
                            oskar_vis_block_cross_correlations_const(block),
                            num_baselines * t,
                            num_baselines * (num_channels * t + c),
                            num_baselines, status);
                }
                oskar_timer_pause(h->tmr_copy_convert);
                oskar_imager_update(h, num_rows,
                        start_chan + c, start_chan + c, num_pols,
                        oskar_vis_block_baseline_uu_metres_const(block),
                        oskar_vis_block_baseline_vv_metres_const(block),
                        oskar_vis_block_baseline_ww_metres_const(block),
                        scratch, weight_ptr, time_centroid, status);
            }
        }
        oskar_mem_free(scratch, status);
    }
    else
    {
        for (c = 0; c < num_channels; ++c)
        {
            /* Update per channel. */
            const double freq_hz =
                    freq_start_hz + (start_chan + c) * freq_inc_hz;
            if (freq_hz >= h->freq_min_hz &&
                    (freq_hz <= h->freq_max_hz || h->freq_max_hz == 0.0))
            {
                oskar_imager_update(h, num_rows,
                        start_chan + c, start_chan + c, num_pols,
                        oskar_vis_block_baseline_uu_metres_const(block),
                        oskar_vis_block_baseline_vv_metres_const(block),
                        oskar_vis_block_baseline_ww_metres_const(block),
                        0, weight_ptr, time_centroid, status);
            }
        }
    }

    oskar_mem_free(weight, status);
    oskar_mem_free(time_centroid, status);
}

#if 0
static int compare_f(const void *p0, const void *p1)
{
    float w0 = fabs(**(const float* const*)p0);
    float w1 = fabs(**(const float* const*)p1);
    if (w0 > w1) return 1;
    if (w0 < w1) return -1;
    return 0;
}

static int compare_d(const void *p0, const void *p1)
{
    double w0 = fabs(**(const double* const*)p0);
    double w1 = fabs(**(const double* const*)p1);
    if (w0 > w1) return 1;
    if (w0 < w1) return -1;
    return 0;
}

static void oskar_imager_sort_by_abs_w(size_t num_vis,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, oskar_Mem* amps,
        oskar_Mem* weight, int* status)
{
    if (*status || num_vis == 0) return;
    /* Based on:
     * https://stackoverflow.com/questions/32948281/c-sort-two-arrays-the-same-way */
    if (oskar_mem_precision(weight) == OSKAR_SINGLE)
    {
        size_t i = 0;
        float** ptr_w = (float**) calloc(num_vis, sizeof(float*));
        float* uu_ = oskar_mem_float(uu, status);
        float* vv_ = oskar_mem_float(vv, status);
        float* ww_ = oskar_mem_float(ww, status);
        float2* vis_ = oskar_mem_float2(amps, status);
        float* weight_ = oskar_mem_float(weight, status);
        for (i = 0; i < num_vis; ++i) ptr_w[i] = &ww_[i];
        qsort(ptr_w, num_vis, sizeof(void*), compare_f);
        for (i = 0; i < num_vis; ++i)
        {
            if (i != (size_t) (ptr_w[i] - ww_))
            {
                size_t j, k;
                const float temp_u = uu_[i];
                const float temp_v = vv_[i];
                const float temp_w = ww_[i];
                const float2 temp_vis = vis_[i];
                const float temp_weight = weight_[i];
                k = i;
                while (i != (j = (size_t) (ptr_w[k] - ww_)))
                {
                    uu_[k] = uu_[j];
                    vv_[k] = vv_[j];
                    ww_[k] = ww_[j];
                    vis_[k] = vis_[j];
                    weight_[k] = weight_[j];
                    ptr_w[k] = &ww_[k];
                    k = j;
                }
                uu_[k] = temp_u;
                vv_[k] = temp_v;
                ww_[k] = temp_w;
                vis_[k] = temp_vis;
                weight_[k] = temp_weight;
                ptr_w[k] = &ww_[k];
            }
        }
        free(ptr_w);
    }
    else
    {
        size_t i = 0;
        double** ptr_w = (double**) calloc(num_vis, sizeof(double*));
        double* uu_ = oskar_mem_double(uu, status);
        double* vv_ = oskar_mem_double(vv, status);
        double* ww_ = oskar_mem_double(ww, status);
        double2* vis_ = oskar_mem_double2(amps, status);
        double* weight_ = oskar_mem_double(weight, status);
        for (i = 0; i < num_vis; ++i) ptr_w[i] = &ww_[i];
        qsort(ptr_w, num_vis, sizeof(void*), compare_d);
        for (i = 0; i < num_vis; ++i)
        {
            if (i != (size_t) (ptr_w[i] - ww_))
            {
                size_t j, k;
                const double temp_u = uu_[i];
                const double temp_v = vv_[i];
                const double temp_w = ww_[i];
                const double2 temp_vis = vis_[i];
                const double temp_weight = weight_[i];
                k = i;
                while (i != (j = (size_t) (ptr_w[k] - ww_)))
                {
                    uu_[k] = uu_[j];
                    vv_[k] = vv_[j];
                    ww_[k] = ww_[j];
                    vis_[k] = vis_[j];
                    weight_[k] = weight_[j];
                    ptr_w[k] = &ww_[k];
                    k = j;
                }
                uu_[k] = temp_u;
                vv_[k] = temp_v;
                ww_[k] = temp_w;
                vis_[k] = temp_vis;
                weight_[k] = temp_weight;
                ptr_w[k] = &ww_[k];
            }
        }
        free(ptr_w);
    }
}
#endif

void oskar_imager_update(oskar_Imager* h, size_t num_rows, int start_chan,
        int end_chan, int num_pols, const oskar_Mem* uu, const oskar_Mem* vv,
        const oskar_Mem* ww, const oskar_Mem* amps, const oskar_Mem* weight,
        const oskar_Mem* time_centroid, int* status)
{
    int c = 0, p = 0, i_plane = 0;
    size_t max_num_vis = 0;
    oskar_Mem *tu = 0, *tv = 0, *tw = 0, *ta = 0, *th = 0;
    const oskar_Mem *u_in = 0, *v_in = 0, *w_in = 0, *amp_in = 0, *weight_in = 0;
    if (*status) return;

    /* Set dimensions. */
    if (num_rows == 0)
    {
        num_rows = oskar_mem_length(uu);
    }

    /* Check polarisation type. */
    if (num_pols == 1 && h->im_type != OSKAR_IMAGE_TYPE_I &&
            h->im_type != OSKAR_IMAGE_TYPE_PSF)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Ensure image/grid planes exist and algorithm has been initialised. */
    oskar_imager_set_num_planes(h, status);
    oskar_imager_check_init(h, status);
    oskar_imager_allocate_planes(h, status);
    if (*status) return;

    /* Convert precision of input data if required. */
    u_in = uu; v_in = vv; w_in = ww; weight_in = weight;
    if (!h->coords_only)
    {
        if (!amps)
        {
            *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
            return;
        }
        amp_in = amps;
        if (oskar_mem_precision(amps) != h->imager_prec)
        {
            oskar_timer_resume(h->tmr_copy_convert);
            ta = oskar_mem_convert_precision(amps, h->imager_prec, status);
            oskar_timer_pause(h->tmr_copy_convert);
            amp_in = ta;
        }

        /* Convert linear polarisations to Stokes parameters if required. */
        if (h->use_stokes)
        {
            oskar_timer_resume(h->tmr_copy_convert);
            oskar_imager_linear_to_stokes(amp_in, &h->stokes, status);
            oskar_timer_pause(h->tmr_copy_convert);
            amp_in = h->stokes;
        }
    }
    if (oskar_mem_precision(uu) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        tu = oskar_mem_convert_precision(uu, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        u_in = tu;
    }
    if (oskar_mem_precision(vv) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        tv = oskar_mem_convert_precision(vv, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        v_in = tv;
    }
    if (oskar_mem_precision(ww) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        tw = oskar_mem_convert_precision(ww, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        w_in = tw;
    }
    if (oskar_mem_precision(weight) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        th = oskar_mem_convert_precision(weight, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        weight_in = th;
    }

    /* Ensure work arrays are large enough. */
    max_num_vis = num_rows;
    if (!h->chan_snaps) max_num_vis *= (1 + end_chan - start_chan);
    oskar_mem_ensure(h->uu_im, max_num_vis, status);
    oskar_mem_ensure(h->vv_im, max_num_vis, status);
    oskar_mem_ensure(h->ww_im, max_num_vis, status);
    if (!h->coords_only)
    {
        oskar_mem_ensure(h->vis_im, max_num_vis, status);
    }
    oskar_mem_ensure(h->weight_im, max_num_vis, status);
    if (h->direction_type == 'R')
    {
        oskar_mem_ensure(h->uu_tmp, max_num_vis, status);
        oskar_mem_ensure(h->vv_tmp, max_num_vis, status);
        oskar_mem_ensure(h->ww_tmp, max_num_vis, status);
    }

    /* Loop over each image plane being made. */
    for (c = 0; c < h->num_im_channels; ++c)
    {
        for (p = 0; p < h->num_im_pols; ++p)
        {
            oskar_Mem *pu = 0, *pv = 0, *pw = 0, *pt = 0;
            size_t num_vis = 0;
            if (*status) break;

            /* Get all visibility data needed to update this plane. */
            pu = h->uu_im; pv = h->vv_im; pw = h->ww_im; pt = h->time_im;
            if (h->direction_type == 'R')
            {
                pu = h->uu_tmp; pv = h->vv_tmp; pw = h->ww_tmp;
            }
            if (h->time_min_utc <= 0.0 && h->time_max_utc <= 0.0) pt = 0;
            oskar_timer_resume(h->tmr_select_scale);
            oskar_imager_select_data(h, num_rows, start_chan, end_chan,
                    num_pols, u_in, v_in, w_in, amp_in, weight_in,
                    time_centroid, h->im_freqs[c], p,
                    &num_vis, pu, pv, pw, h->vis_im, h->weight_im,
                    pt, status);
            oskar_timer_pause(h->tmr_select_scale);

            /* Skip if nothing was selected. */
            if (num_vis == 0) continue;

            /* Rotate baseline coordinates if required. */
            if (h->direction_type == 'R')
            {
                oskar_imager_rotate_coords(h, num_vis,
                        h->uu_tmp, h->vv_tmp, h->ww_tmp,
                        h->uu_im, h->vv_im, h->ww_im);
            }

            /* Overwrite visibilities if making PSF, or phase rotate. */
            if (!h->coords_only)
            {
                if (h->im_type == OSKAR_IMAGE_TYPE_PSF)
                {
                    oskar_mem_set_value_real(h->vis_im, 1.0,
                            0, oskar_mem_length(h->vis_im), status);
                }
                else if (h->direction_type == 'R')
                {
                    oskar_imager_rotate_vis(h, num_vis,
                            h->uu_tmp, h->vv_tmp, h->ww_tmp, h->vis_im);
                }
            }

            /* Apply time and baseline length filters if required. */
            oskar_imager_filter_time(h, &num_vis, h->uu_im, h->vv_im,
                    h->ww_im, h->vis_im, h->weight_im, pt, status);
            oskar_imager_filter_uv(h, &num_vis, h->uu_im, h->vv_im,
                    h->ww_im, h->vis_im, h->weight_im, status);

#if 0
            /* Sort visibility data by w coordinate. */
            if (h->algorithm == OSKAR_ALGORITHM_WPROJ && !h->coords_only)
                oskar_imager_sort_by_abs_w(num_vis, h->uu_im, h->vv_im,
                        h->ww_im, h->vis_im, h->weight_im, status);
#endif

            /* Update this image plane with the visibilities. */
            i_plane = h->num_im_pols * c + p;
            oskar_imager_update_plane(h, num_vis, h->uu_im, h->vv_im,
                    h->ww_im, (h->coords_only ? 0 : h->vis_im), h->weight_im,
                    i_plane, 0, 0, h->weights_grids[i_plane], status);
        }
    }

    oskar_mem_free(tu, status);
    oskar_mem_free(tv, status);
    oskar_mem_free(tw, status);
    oskar_mem_free(ta, status);
    oskar_mem_free(th, status);
}


void oskar_imager_update_plane(oskar_Imager* h, size_t num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, int i_plane,
        oskar_Mem* plane, double* plane_norm, oskar_Mem* weights_grid,
        int* status)
{
    oskar_Mem *tu = 0, *tv = 0, *tw = 0, *ta = 0, *th = 0;
    const oskar_Mem *pu = 0, *pv = 0, *pw = 0, *pa = 0, *ph = 0;
    if (*status || num_vis == 0) return;

    /* Convert precision of input data if required. */
    pu = uu; pv = vv; pw = ww; ph = weight;
    if (oskar_mem_precision(uu) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        tu = oskar_mem_convert_precision(uu, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        pu = tu;
    }
    if (oskar_mem_precision(vv) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        tv = oskar_mem_convert_precision(vv, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        pv = tv;
    }
    if (oskar_mem_precision(ww) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        tw = oskar_mem_convert_precision(ww, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        pw = tw;
    }
    if (oskar_mem_precision(weight) != h->imager_prec)
    {
        oskar_timer_resume(h->tmr_copy_convert);
        th = oskar_mem_convert_precision(weight, h->imager_prec, status);
        oskar_timer_pause(h->tmr_copy_convert);
        ph = th;
    }

    /* Just update the grid of weights if we're in coordinate-only mode. */
    if (h->coords_only)
    {
        oskar_imager_update_weights_grid(h, num_vis, pu, pv, pw, ph,
                weights_grid, h->weights_guard[i_plane], status);
    }
    else
    {
        size_t num_skipped = 0;
        double* plane_norm_ptr = plane_norm;

        /* Convert precision of visibility amplitudes if required. */
        pa = amps;
        if (oskar_mem_precision(amps) != h->imager_prec)
        {
            oskar_timer_resume(h->tmr_copy_convert);
            ta = oskar_mem_convert_precision(amps, h->imager_prec, status);
            oskar_timer_pause(h->tmr_copy_convert);
            pa = ta;
        }

        /* Check imager is ready. */
        oskar_imager_check_init(h, status);

        /* Re-weight visibilities if required. */
        switch (h->weighting)
        {
        case OSKAR_WEIGHTING_NATURAL:
            /* Nothing to do. */
            break;
        case OSKAR_WEIGHTING_RADIAL:
            oskar_timer_resume(h->tmr_weights_lookup);
            oskar_imager_weight_radial(num_vis, pu, pv, ph, h->weight_tmp,
                    status);
            oskar_timer_pause(h->tmr_weights_lookup);
            ph = h->weight_tmp;
            break;
        case OSKAR_WEIGHTING_UNIFORM:
            oskar_timer_resume(h->tmr_weights_lookup);
            oskar_imager_weight_uniform(num_vis, pu, pv, ph, h->weight_tmp,
                    h->cellsize_rad, oskar_imager_plane_size(h), weights_grid,
                    &num_skipped, status);
            oskar_timer_pause(h->tmr_weights_lookup);
            ph = h->weight_tmp;
            break;
        default:
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            break;
        }
        if (num_skipped > 0)
        {
            oskar_log_warning(h->log, "Skipped %lu visibility weights.",
                    (unsigned long) num_skipped);
        }

        /* Apply (u,v) tapering to weights if required. */
        if (h->uv_taper[0] > 0.0 || h->uv_taper[1] > 0.0)
        {
            oskar_imager_taper_weights(num_vis, pu, pv, ph, h->weight_tmp,
                    h->uv_taper, status);
            ph = h->weight_tmp;
        }

        /* Update the supplied plane with the supplied visibilities. */
        num_skipped = 0;
        if (!plane_norm_ptr && h->plane_norm)
        {
            plane_norm_ptr = &(h->plane_norm[i_plane]);
        }
        oskar_timer_resume(h->tmr_grid_update);
        switch (h->algorithm)
        {
        case OSKAR_ALGORITHM_DFT_2D:
        case OSKAR_ALGORITHM_DFT_3D:
            oskar_imager_update_plane_dft(h, num_vis, pu, pv, pw, pa, ph,
                    i_plane, plane, plane_norm_ptr, status);
            break;
        case OSKAR_ALGORITHM_FFT:
            oskar_imager_update_plane_fft(h, num_vis, pu, pv, pa, ph,
                    i_plane, plane, plane_norm_ptr, &num_skipped, status);
            break;
        case OSKAR_ALGORITHM_WPROJ:
            oskar_imager_update_plane_wproj(h, num_vis, pu, pv, pw, pa, ph,
                    i_plane, plane, plane_norm_ptr, &num_skipped, status);
            break;
        default:
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            break;
        }
        oskar_timer_pause(h->tmr_grid_update);
        h->num_vis_processed += (num_vis - num_skipped);
        if (num_skipped > 0)
        {
            oskar_log_warning(h->log, "Skipped %lu visibility points.",
                    (unsigned long) num_skipped);
        }
    }

    /* Clean up. */
    oskar_mem_free(tu, status);
    oskar_mem_free(tv, status);
    oskar_mem_free(tw, status);
    oskar_mem_free(ta, status);
    oskar_mem_free(th, status);
}


void oskar_imager_update_weights_grid(oskar_Imager* h, size_t num_points,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* weight, oskar_Mem* weights_grid,
        oskar_Mem* weights_guard, int* status)
{
    if (*status) return;

    /* Update the weights grid. */
    if (h->weighting == OSKAR_WEIGHTING_UNIFORM)
    {
        size_t num_cells = 0, num_skipped = 0;

        /* Resize the grid of weights if needed. */
        const int grid_size = oskar_imager_plane_size(h);
        num_cells = (size_t) grid_size * (size_t) grid_size;
        oskar_mem_ensure(weights_grid, num_cells, status);
        if (*status) return;

        oskar_timer_resume(h->tmr_weights_grid);
        if (oskar_mem_precision(weights_grid) == OSKAR_DOUBLE)
        {
            oskar_grid_weights_write_d(num_points,
                    oskar_mem_double_const(uu, status),
                    oskar_mem_double_const(vv, status),
                    oskar_mem_double_const(weight, status),
                    h->cellsize_rad, grid_size, &num_skipped,
                    oskar_mem_double(weights_grid, status));
        }
        else
        {
            oskar_mem_ensure(weights_guard, num_cells, status);
            if (*status) return;
            oskar_grid_weights_write_f(num_points,
                    oskar_mem_float_const(uu, status),
                    oskar_mem_float_const(vv, status),
                    oskar_mem_float_const(weight, status),
                    (float) (h->cellsize_rad), grid_size, &num_skipped,
                    oskar_mem_float(weights_grid, status),
                    oskar_mem_float(weights_guard, status));
        }
        if (num_skipped > 0)
        {
            oskar_log_warning(h->log, "Skipped %lu visibility weights.",
                    (unsigned long) num_skipped);
        }
        oskar_timer_pause(h->tmr_weights_grid);
    }

    /* Update baseline W minimum, maximum and RMS. */
    if (h->algorithm == OSKAR_ALGORITHM_WPROJ)
    {
        size_t j = 0;
        oskar_timer_resume(h->tmr_coord_scan);
        if (oskar_mem_precision(ww) == OSKAR_DOUBLE)
        {
            const double *p = oskar_mem_double_const(ww, status);
            for (j = 0; j < num_points; ++j)
            {
                const double val = fabs(p[j]);
                h->ww_rms += (val * val);
                if (val < h->ww_min) h->ww_min = val;
                if (val > h->ww_max) h->ww_max = val;
            }
        }
        else
        {
            const float *p = oskar_mem_float_const(ww, status);
            for (j = 0; j < num_points; ++j)
            {
                const double val = fabs((double) (p[j]));
                h->ww_rms += (val * val);
                if (val < h->ww_min) h->ww_min = val;
                if (val > h->ww_max) h->ww_max = val;
            }
        }
        h->ww_points += num_points;
        oskar_timer_pause(h->tmr_coord_scan);
    }
}

void oskar_imager_allocate_planes(oskar_Imager* h, int *status)
{
    int i = 0;
    if (*status) return;

    /* Don't continue if we're in "coords only" mode or if planes are
     * already allocated. */
    if (h->coords_only || h->planes) return;

    /* Record the plane size. */
    const int num_planes = h->num_planes;
    const int plane_size = oskar_imager_plane_size(h);
    const int plane_type = oskar_imager_plane_type(h);
    const size_t num_cells = ((size_t) plane_size) * ((size_t) plane_size);
    const size_t plane_mem = num_cells * oskar_mem_element_size(plane_type);
    oskar_log_message(h->log, 'M', 0, "Plane size is %d x %d.",
            plane_size, plane_size);
    oskar_log_message(h->log, 'M', 0, "Allocating %d plane(s) of size "
            "%.1f MB (%.1f MB total).", num_planes, plane_mem * 1e-6,
            num_planes * plane_mem * 1e-6);

    /* Allocate the image or visibility planes on the host. */
    h->planes = (oskar_Mem**) calloc(num_planes, sizeof(oskar_Mem*));
    h->plane_norm = (double*) calloc(num_planes, sizeof(double));
    for (i = 0; i < num_planes; ++i)
    {
        h->planes[i] = oskar_mem_create(plane_type, OSKAR_CPU,
                num_cells, status);
    }

    /* Allocate visibility planes on the devices if required. */
    if (h->grid_on_gpu && !(
            h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
            h->algorithm == OSKAR_ALGORITHM_DFT_3D))
    {
        int j = 0, norm_type = 0;
        const int loc = h->dev_loc;
        for (j = 0; j < h->num_gpus; ++j)
        {
            if (*status) break;
            DeviceData* d = &h->d[j];
            d->num_planes = num_planes;
            d->planes = (oskar_Mem**) calloc(num_planes, sizeof(oskar_Mem*));
            oskar_log_message(h->log, 'M', 0,
                    "Allocating memory on device %d for visibility grids.",
                    h->gpu_ids[j]);
            oskar_device_set(loc, h->gpu_ids[j], status);
            for (i = 0; i < num_planes; ++i)
            {
                d->planes[i] = oskar_mem_create(plane_type, loc,
                        num_cells, status);
                oskar_mem_clear_contents(d->planes[i], status);
            }

            /* Get the normalisation type. */
            if (oskar_device_supports_double(loc) &&
                    oskar_device_supports_atomic64(loc))
            {
                norm_type = OSKAR_DOUBLE;
            }
            else
            {
                norm_type = OSKAR_SINGLE;
            }

            /* Define (empty) device arrays for scratch data. */
            d->uu = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->vv = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->ww = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->vis = oskar_mem_create(plane_type, loc, 0, status);
            d->weight = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->counter = oskar_mem_create(OSKAR_INT, loc, 1, status);
            d->count_skipped = oskar_mem_create(OSKAR_INT, loc, 1, status);
            d->norm = oskar_mem_create(norm_type, loc, 1, status);
            d->num_points_in_tiles =
                    oskar_mem_create(OSKAR_INT, loc, 0, status);
            d->tile_offsets = oskar_mem_create(OSKAR_INT, loc, 0, status);
            d->tile_locks = oskar_mem_create(OSKAR_INT, loc, 0, status);
            d->sorted_uu = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->sorted_vv = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->sorted_ww = oskar_mem_create(OSKAR_INT, loc, 0, status);
            d->sorted_wt = oskar_mem_create(h->imager_prec, loc, 0, status);
            d->sorted_vis = oskar_mem_create(plane_type, loc, 0, status);
            d->sorted_tile = oskar_mem_create(OSKAR_INT, loc, 0, status);
        }
    }

    /* Create FITS files for the planes if required. */
    oskar_imager_create_fits_files(h, status);
}


#ifdef __cplusplus
}
#endif
