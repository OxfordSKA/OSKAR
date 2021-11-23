/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager_reset_cache.h"
#include "imager/private_imager_free_device_data.h"
#include "log/oskar_log.h"
#include "math/oskar_fft.h"
#include <fitsio.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_reset_cache(oskar_Imager* h, int* status)
{
    int i = 0;

    /* Clear all device data. */
    oskar_imager_free_device_data(h, status);

    /* Clear selected axes. */
    free(h->sel_freqs); h->sel_freqs = 0;
    free(h->im_freqs); h->im_freqs = 0;
    h->num_sel_freqs = 0;
    h->num_im_channels = 0;

    /* Clear FFT caches. */
    oskar_fft_free(h->fft); h->fft = 0;
    oskar_mem_free(h->corr_func, status); h->corr_func = 0;

    /* Clear algorithm-specific caches. */
    oskar_mem_free(h->l, status); h->l = 0;
    oskar_mem_free(h->m, status); h->m = 0;
    oskar_mem_free(h->n, status); h->n = 0;
    oskar_mem_free(h->conv_func, status); h->conv_func = 0;
    oskar_mem_free(h->w_support, status); h->w_support = 0;
    oskar_mem_free(h->w_kernels_compact, status); h->w_kernels_compact = 0;
    oskar_mem_free(h->w_kernel_start, status); h->w_kernel_start = 0;

    /* Free the image planes. */
    if (h->planes)
    {
        for (i = 0; i < h->num_planes; ++i)
        {
            oskar_mem_free(h->planes[i], status);
        }
    }
    free(h->planes); h->planes = 0;
    free(h->plane_norm); h->plane_norm = 0;

    /* Free the weights grids if they exist. */
    if (h->weights_grids)
    {
        for (i = 0; i < h->num_planes; ++i)
        {
            oskar_mem_free(h->weights_grids[i], status);
            oskar_mem_free(h->weights_guard[i], status);
        }
    }
    free(h->weights_grids); h->weights_grids = 0;
    free(h->weights_guard); h->weights_guard = 0;

    /* Collapse temp arrays. */
    oskar_mem_realloc(h->uu_im, 0, status);
    oskar_mem_realloc(h->vv_im, 0, status);
    oskar_mem_realloc(h->ww_im, 0, status);
    oskar_mem_realloc(h->uu_tmp, 0, status);
    oskar_mem_realloc(h->vv_tmp, 0, status);
    oskar_mem_realloc(h->ww_tmp, 0, status);
    oskar_mem_realloc(h->vis_im, 0, status);
    oskar_mem_realloc(h->weight_im, 0, status);
    oskar_mem_realloc(h->weight_tmp, 0, status);
    oskar_mem_realloc(h->time_im, 0, status);
    oskar_mem_free(h->stokes, status); h->stokes = 0;

    /* Close any open FITS files. */
    for (i = 0; i < h->num_im_pols; ++i)
    {
        if (h->fits_file[i])
        {
            ffclos(h->fits_file[i], status);
        }
        h->fits_file[i] = 0;
        free(h->output_name[i]); h->output_name[i] = 0;
    }

    /* Clear the number of image planes. */
    h->num_planes = 0;

    /* Clear the timers. */
    oskar_timer_reset(h->tmr_grid_finalise);
    oskar_timer_reset(h->tmr_grid_update);
    oskar_timer_reset(h->tmr_init);
    oskar_timer_reset(h->tmr_select_scale);
    oskar_timer_reset(h->tmr_filter);
    oskar_timer_reset(h->tmr_read);
    oskar_timer_reset(h->tmr_write);
    oskar_timer_start(h->tmr_overall);
    oskar_timer_reset(h->tmr_copy_convert);
    oskar_timer_reset(h->tmr_coord_scan);
    oskar_timer_reset(h->tmr_rotate);
    oskar_timer_reset(h->tmr_weights_grid);
    oskar_timer_reset(h->tmr_weights_lookup);

    /* Clear state. */
    h->init = 0;
    h->num_vis_processed = 0;
}

#ifdef __cplusplus
}
#endif
