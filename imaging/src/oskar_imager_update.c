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

#include <private_imager.h>

#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_convert_lon_lat_to_relative_directions.h>
#include <oskar_imager.h>
#include <private_imager_algorithm_init_dft.h>
#include <private_imager_algorithm_init_fft.h>
#include <private_imager_algorithm_init_wproj.h>
#include <private_imager_create_fits_files.h>
#include <private_imager_update_plane_dft.h>
#include <private_imager_update_plane_fft.h>

#include <stdlib.h>
#include <stdio.h>

#define DEG2RAD M_PI/180.0
#define SEC2DAYS 1.15740740740740740740741e-5

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_imager_allocate_image_planes(oskar_Imager* h, int *status);

void oskar_imager_update(oskar_Imager* h, int start_time, int end_time,
        int start_chan, int end_chan, int num_pols, int num_baselines,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, int* status)
{
    int t, c, p, plane, num_times, num_channels, max_num_vis;
    oskar_Mem *tu = 0, *tv = 0, *tw = 0, *ta = 0;
    const oskar_Mem *data, *u_in, *v_in, *w_in, *amp_in;
    oskar_Mem *pu, *pv, *pw;

    /* Set dimensions. */
    num_times = 1 + end_time - start_time;
    num_channels = 1 + end_chan - start_chan;

    /* Check polarisation type. */
    if (num_pols == 1 && h->im_type != OSKAR_IMAGE_TYPE_I &&
            h->im_type != OSKAR_IMAGE_TYPE_PSF)
    {
        *status = OSKAR_ERR_SETTINGS_IMAGE;
        return;
    }

    /* Ensure post-initialisation steps have been done. */
    if (!h->planes)
    {
        oskar_imager_allocate_image_planes(h, status);
        oskar_imager_create_fits_files(h, status);
        if (*status) return;
    }

    /* Convert precision of input data if required. */
    u_in = uu; v_in = vv; w_in = ww; amp_in = amps;
    if (oskar_mem_precision(uu) != h->imager_prec)
    {
        tu = oskar_mem_convert_precision(uu, h->imager_prec, status);
        u_in = tu;
    }
    if (oskar_mem_precision(vv) != h->imager_prec)
    {
        tv = oskar_mem_convert_precision(vv, h->imager_prec, status);
        v_in = tv;
    }
    if (oskar_mem_precision(ww) != h->imager_prec)
    {
        tw = oskar_mem_convert_precision(ww, h->imager_prec, status);
        w_in = tw;
    }
    if (oskar_mem_precision(amps) != h->imager_prec)
    {
        ta = oskar_mem_convert_precision(amps, h->imager_prec, status);
        amp_in = ta;
    }

    /* Convert linear polarisations to Stokes parameters if required. */
    data = amp_in;
    if (h->use_stokes)
    {
        oskar_imager_linear_to_stokes(amp_in, h->stokes, status);
        data = h->stokes;
    }

    /* Ensure work arrays are large enough. */
    max_num_vis = num_baselines;
    if (h->time_snaps && !h->chan_snaps) /* Frequency synthesis. */
        max_num_vis *= num_channels;
    else if (!h->time_snaps && h->chan_snaps) /* Time synthesis. */
        max_num_vis *= num_times;
    else if (!h->time_snaps && !h->chan_snaps) /* Time & frequency synthesis. */
        max_num_vis *= num_times * num_channels;
    oskar_mem_realloc(h->uu_im, max_num_vis, status);
    oskar_mem_realloc(h->vv_im, max_num_vis, status);
    oskar_mem_realloc(h->ww_im, max_num_vis, status);
    oskar_mem_realloc(h->vis_im, max_num_vis, status);
    if (h->direction_type == 'R')
    {
        oskar_mem_realloc(h->uu_tmp, max_num_vis, status);
        oskar_mem_realloc(h->vv_tmp, max_num_vis, status);
        oskar_mem_realloc(h->ww_tmp, max_num_vis, status);
    }

    /* Loop over each image plane being made. */
    for (t = 0; t < h->im_num_times; ++t)
    {
        if (*status) break;
        for (c = 0; c < h->im_num_channels; ++c)
        {
            size_t num_coords = 0, num_vis = 0;
            if (*status) break;

            /* Get all the baseline coordinates needed to update this plane. */
            pu = h->uu_im; pv = h->vv_im; pw = h->ww_im;
            if (h->direction_type == 'R')
            {
                pu = h->uu_tmp; pv = h->vv_tmp; pw = h->ww_tmp;
            }
            oskar_imager_select_coords(h, start_time, end_time,
                    start_chan, end_chan, num_baselines, u_in, v_in, w_in,
                    t, c, pu, pv, pw, &num_coords, status);

            /* Check if any baselines were selected. */
            if (num_coords == 0) continue;

            /* Rotate baseline coordinates if required. */
            if (h->direction_type == 'R')
                oskar_imager_rotate_coords((int)num_coords,
                        h->uu_tmp, h->vv_tmp, h->ww_tmp, h->M,
                        h->uu_im, h->vv_im, h->ww_im);

            for (p = 0; p < h->im_num_pols; ++p)
            {
                if (*status) break;

                /* Get visibility amplitudes for imaging. */
                if (h->im_type == OSKAR_IMAGE_TYPE_PSF)
                {
                    oskar_mem_set_value_real(h->vis_im, 1.0, 0, 0, status);
                    num_vis = num_coords;
                }
                else
                {
                    oskar_imager_select_vis(h,
                            start_time, end_time, start_chan, end_chan,
                            num_baselines, num_pols, data, t, c, p,
                            h->vis_im, &num_vis, status);

                    /* Phase rotate the visibilities if required. */
                    if (h->direction_type == 'R')
                        oskar_imager_rotate_vis(num_vis,
                                h->uu_tmp, h->vv_tmp, h->ww_tmp, h->vis_im,
                                h->delta_l, h->delta_m, h->delta_n);
                }

                /* Check consistency. */
                if (num_coords != num_vis)
                {
                    fprintf(stderr, "Internal error: Inconsistent number of "
                            "baseline coordinates and visibility "
                            "amplitudes.\n");
                    exit(1);
                }

                /* Update this image plane with the visibilities. */
                plane = h->im_num_pols * (t * h->im_num_channels + c) + p;
                oskar_imager_update_plane(h, num_vis, h->uu_im, h->vv_im,
                        h->ww_im, h->vis_im, h->planes[plane],
                        &h->plane_norm[plane], status);
            } /* End image pol */
        } /* End image channel */
    } /* End image time */

    oskar_mem_free(tu, status);
    oskar_mem_free(tv, status);
    oskar_mem_free(tw, status);
    oskar_mem_free(ta, status);
}


void oskar_imager_update_plane(oskar_Imager* h, int num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, oskar_Mem* plane, double* plane_norm,
        int* status)
{
    oskar_Mem *tu = 0, *tv = 0, *tw = 0, *ta = 0;
    const oskar_Mem *pu, *pv, *pw, *pa;
    if (oskar_mem_precision(plane) != h->imager_prec)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Convert precision of input data if required. */
    pu = uu; pv = vv; pw = ww; pa = amps;
    if (oskar_mem_precision(uu) != h->imager_prec)
    {
        tu = oskar_mem_convert_precision(uu, h->imager_prec, status);
        pu = tu;
    }
    if (oskar_mem_precision(vv) != h->imager_prec)
    {
        tv = oskar_mem_convert_precision(vv, h->imager_prec, status);
        pv = tv;
    }
    if (oskar_mem_precision(ww) != h->imager_prec)
    {
        tw = oskar_mem_convert_precision(ww, h->imager_prec, status);
        pw = tw;
    }
    if (oskar_mem_precision(amps) != h->imager_prec)
    {
        ta = oskar_mem_convert_precision(amps, h->imager_prec, status);
        pa = ta;
    }

    /* Update the supplied plane with the supplied visibilities. */
    if (h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
            h->algorithm == OSKAR_ALGORITHM_DFT_3D)
    {
        if (!h->l) oskar_imager_algorithm_init_dft(h, status);
        *plane_norm += (double) num_vis;
        oskar_imager_update_plane_dft(h, num_vis, pu, pv, pw, pa,
                plane, status);
    }
    else if (h->algorithm == OSKAR_ALGORITHM_FFT)
    {
        if (!h->conv_func) oskar_imager_algorithm_init_fft(h, status);
        oskar_imager_update_plane_fft(h, num_vis, pu, pv, pa,
                plane, plane_norm, status);
    }
    else if (h->algorithm == OSKAR_ALGORITHM_WPROJ)
    {
        /* if (!h->w_kernels) oskar_imager_algorithm_init_wproj(h, status); */
        /* *plane_norm += (double) num_vis; */
        /* oskar_imager_update_plane_wproj(h, num_vis, pu, pv, pw, pa,
                plane, status); */
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
    else
    {
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
    oskar_mem_free(tu, status);
    oskar_mem_free(tv, status);
    oskar_mem_free(tw, status);
    oskar_mem_free(ta, status);
}


void oskar_imager_allocate_image_planes(oskar_Imager* h, int *status)
{
    int i;
    if (*status) return;

    /* Set image meta-data. */
    h->im_num_channels = (h->chan_snaps ?
            1 + h->vis_chan_range[1] - h->vis_chan_range[0] : 1);
    h->im_num_times = (h->time_snaps ?
            1 + h->vis_time_range[1] - h->vis_time_range[0] : 1);
    h->im_time_start_mjd_utc = h->vis_time_start_mjd_utc +
            (h->vis_time_range[0] * h->time_inc_sec * SEC2DAYS);
    if (h->chan_snaps)
    {
        h->im_freq_start_hz = h->vis_freq_start_hz +
                h->vis_chan_range[0] * h->freq_inc_hz;
    }
    else
    {
        double chan0 = 0.5 * (h->vis_chan_range[1] - h->vis_chan_range[0]);
        h->im_freq_start_hz = h->vis_freq_start_hz + chan0 * h->freq_inc_hz;
    }
    if (h->direction_type != 'R')
    {
        h->im_centre_deg[0] = h->vis_centre_deg[0];
        h->im_centre_deg[1] = h->vis_centre_deg[1];
    }

    /*************************************************************************/
    /* Allocate the image or visibility planes. */
    oskar_imager_reset_cache(h, status);
    h->num_planes = h->im_num_times * h->im_num_channels * h->im_num_pols;
    h->planes = calloc(h->num_planes, sizeof(oskar_Mem*));
    h->plane_tmp = oskar_mem_create(h->imager_prec, OSKAR_CPU,
            h->num_pixels, status);
    for (i = 0; i < h->num_planes; ++i)
        if (h->algorithm == OSKAR_ALGORITHM_FFT)
            h->planes[i] = oskar_mem_create(h->imager_prec | OSKAR_COMPLEX,
                    OSKAR_CPU, h->num_pixels, status);
        else
            h->planes[i] = oskar_mem_create(h->imager_prec,
                    OSKAR_CPU, h->num_pixels, status);
    if (*status) return;
    h->plane_norm = calloc(h->num_planes, sizeof(double));

    /* If imaging away from the beam direction, evaluate l0-l, m0-m, n0-n
     * for the new pointing centre, and a rotation matrix to generate the
     * rotated baseline coordinates. */
    if (h->direction_type == 'R')
    {
        double l1, m1, n1, ra_rad, dec_rad, ra0_rad, dec0_rad;
        double d_a, d_d, *M;

        ra_rad = h->im_centre_deg[0] * DEG2RAD;
        dec_rad = h->im_centre_deg[1] * DEG2RAD;
        ra0_rad = h->vis_centre_deg[0] * DEG2RAD;
        dec0_rad = h->vis_centre_deg[1] * DEG2RAD;
        d_a = ra0_rad - ra_rad; /* It's OK, these are meant to be swapped. */
        d_d = dec_rad - dec0_rad;

        /* Rotate by -delta_ra around v, then delta_dec around u. */
        M = h->M;
        M[0] = cos(d_a);           M[1] = 0.0;      M[2] = sin(d_a);
        M[3] = sin(d_a)*sin(d_d);  M[4] = cos(d_d); M[5] = -cos(d_a)*sin(d_d);
        M[6] = -sin(d_a)*cos(d_d); M[7] = sin(d_a); M[8] = cos(d_a)*cos(d_d);

        oskar_convert_lon_lat_to_relative_directions_d(1,
                &ra_rad, &dec_rad, ra0_rad, dec0_rad, &l1, &m1, &n1);
        h->delta_l = 0 - l1;
        h->delta_m = 0 - m1;
        h->delta_n = 1 - n1;
    }
}


#ifdef __cplusplus
}
#endif
