/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/oskar_grid_correction.h"
#include "imager/oskar_grid_functions_pillbox.h"
#include "imager/oskar_grid_functions_spheroidal.h"
#include "imager/private_imager_free_device_data.h"
#include "math/oskar_fft.h"
#include "math/oskar_fftphase.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_get_memory_usage.h"
#include "utility/oskar_timer.h"

#include <fitsio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void write_plane(oskar_Imager* h, oskar_Mem* plane,
        int c, int p, int* status);


void oskar_imager_finalise(oskar_Imager* h,
        int num_output_images, oskar_Mem** output_images,
        int num_output_grids, oskar_Mem** output_grids, int* status)
{
    int c = 0, p = 0, i = 0;
    size_t j = 0, log_size = 0, length = 0;
    char* log_data = 0;

    /* Report any error. */
    if (*status)
    {
        oskar_log_error(h->log, "Run failed with code %i: %s.", *status,
                oskar_get_error_string(*status));
        oskar_imager_reset_cache(h, status);
        oskar_log_close(h->log);
        return;
    }

    if (!h->planes) return;
    oskar_log_section(h->log, 'M', "Finalising %d image plane(s)...",
            h->num_planes);

    /* Adjust normalisation if required. */
    if (h->scale_norm_with_num_input_files)
    {
        for (i = 0; i < h->num_planes; ++i)
        {
            h->plane_norm[i] /= h->num_files;
        }
    }

    /* Clear convolution kernels and scratch arrays.
     * Spare device memory may be needed for the FFT plan. */
    oskar_imager_free_device_scratch_data(h, status);

    /* If gridding with multiple GPUs, copy grids to host and combine them. */
    if (h->grid_on_gpu && h->num_gpus > 1 && !(
            h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
            h->algorithm == OSKAR_ALGORITHM_DFT_3D))
    {
        const size_t plane_size = (size_t) oskar_imager_plane_size(h);
        const size_t num_cells = plane_size * plane_size;
        oskar_Mem* temp = oskar_mem_create(oskar_imager_plane_type(h),
                OSKAR_CPU, num_cells, status);
        oskar_log_message(h->log, 'M', 0,
                "Stacking %d grid(s) from %d devices...",
                h->num_planes, h->num_gpus);
        oskar_timer_resume(h->tmr_grid_finalise);
        for (i = 0; i < h->num_planes; ++i)
        {
            int d = 0;
            for (d = 0; d < h->num_gpus; ++d)
            {
                oskar_device_set(h->dev_loc, h->gpu_ids[d], status);
                if (d == 0)
                {
                    oskar_mem_copy(h->planes[i], h->d[d].planes[i], status);
                }
                else
                {
                    oskar_mem_copy(temp, h->d[d].planes[i], status);
                    oskar_mem_add(h->planes[i], h->planes[i], temp,
                            0, 0, 0, num_cells, status);
                }
            }
            oskar_device_set(h->dev_loc, h->gpu_ids[0], status);
            oskar_mem_copy(h->d[0].planes[i], h->planes[i], status);
        }
        oskar_timer_pause(h->tmr_grid_finalise);
        oskar_mem_free(temp, status);
    }

    /* Copy grids to output grid planes if given. */
    for (i = 0; (i < h->num_planes) && (i < num_output_grids); ++i)
    {
        oskar_Mem *plane = h->planes[i];
        if (h->grid_on_gpu && h->num_gpus == 1 && !(
                h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
                h->algorithm == OSKAR_ALGORITHM_DFT_3D))
        {
            plane = h->d[0].planes[i];
        }
        if (!(output_grids[i]))
        {
            output_grids[i] = oskar_mem_create(oskar_mem_type(plane),
                    OSKAR_CPU, 0, status);
        }
        oskar_mem_copy(output_grids[i], plane, status);
        oskar_mem_scale_real(output_grids[i], 1.0 / h->plane_norm[i],
                0, oskar_mem_length(output_grids[i]), status);
    }

    /* Check if images are required. */
    const size_t num_pix = (size_t)h->image_size * (size_t)h->image_size;
    if (h->fits_file[0] || output_images)
    {
        /* Finalise all the planes. */
        for (i = 0; i < h->num_planes; ++i)
        {
            oskar_Mem *plane = h->planes[i];
            if (h->grid_on_gpu && h->num_gpus > 0 && !(
                    h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
                    h->algorithm == OSKAR_ALGORITHM_DFT_3D))
            {
                plane = h->d[0].planes[i];
            }
            oskar_imager_finalise_plane(h, plane, h->plane_norm[i], status);
            if (plane != h->planes[i])
            {
                oskar_mem_copy(h->planes[i], plane, status);
            }
            oskar_imager_trim_image(h, h->planes[i],
                    oskar_imager_plane_size(h), h->image_size, status);
        }

        /* Copy images to output image planes if given. */
        for (i = 0; (i < h->num_planes) && (i < num_output_images); ++i)
        {
            if (!(output_images[i]))
            {
                output_images[i] = oskar_mem_create(h->imager_prec,
                        OSKAR_CPU, num_pix, status);
            }
            oskar_mem_ensure(output_images[i], num_pix, status);
            memcpy(oskar_mem_void(output_images[i]),
                    oskar_mem_void_const(h->planes[i]),
                    num_pix * oskar_mem_element_size(h->imager_prec));
        }

        /* Write to files if required. */
        oskar_timer_resume(h->tmr_write);
        for (c = 0, i = 0; c < h->num_im_channels; ++c)
        {
            for (p = 0; p < h->num_im_pols; ++p, ++i)
            {
                write_plane(h, h->planes[i], c, p, status);
            }
        }
        oskar_timer_pause(h->tmr_write);
    }

    /* Record memory usage. */
    oskar_log_section(h->log, 'M', "Memory usage");
    for (i = 0; i < h->num_gpus; ++i)
    {
        oskar_device_log_mem(h->dev_loc, 0, h->gpu_ids[i], h->log);
    }
    oskar_log_mem(h->log);

    /* Record time taken. */
    oskar_log_set_value_width(h->log, 30);
    oskar_log_section(h->log, 'M', "Imager timing");
    const double t_scan = oskar_timer_elapsed(h->tmr_coord_scan);
    const double t_init = oskar_timer_elapsed(h->tmr_init);
    const double t_copy_convert = oskar_timer_elapsed(h->tmr_copy_convert);
    const double t_select_scale = oskar_timer_elapsed(h->tmr_select_scale);
    const double t_rotate = oskar_timer_elapsed(h->tmr_rotate);
    const double t_filter = oskar_timer_elapsed(h->tmr_filter);
    const double t_grid_update = oskar_timer_elapsed(h->tmr_grid_update);
    const double t_wt_grid = oskar_timer_elapsed(h->tmr_weights_grid);
    const double t_wt_lookup = oskar_timer_elapsed(h->tmr_weights_lookup);
    const double t_grid_finalise = oskar_timer_elapsed(h->tmr_grid_finalise);
    const double t_read = oskar_timer_elapsed(h->tmr_read);
    const double t_write = oskar_timer_elapsed(h->tmr_write);
    if (t_scan > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Coordinate scan", "%.3f s", t_scan);
    }
    if (t_init > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Initialise", "%.3f s", t_init);
    }
    if (t_copy_convert > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Copy/convert data", "%.3f s", t_copy_convert);
    }
    if (t_select_scale > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Select/scale data", "%.3f s", t_select_scale);
    }
    if (t_rotate > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Rotate visibility data", "%.3f s", t_rotate);
    }
    if (t_filter > 1e-3)
    {
        oskar_log_value(h->log, 'M', 0,
            "Filter visibility data", "%.3f s", t_filter);
    }
    if (t_grid_update > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Grid update", "%.3f s", t_grid_update);
    }
    if (t_wt_grid > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Weights grid", "%.3f s", t_wt_grid);
    }
    if (t_wt_lookup > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Weights lookup", "%.3f s", t_wt_lookup);
    }
    if (t_grid_finalise > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Grid finalise", "%.3f s", t_grid_finalise);
    }
    if (t_read > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Read visibility data", "%.3f s", t_read);
    }
    if (t_write > 0.0)
    {
        oskar_log_value(h->log, 'M', 0,
            "Write image data", "%.3f s", t_write);
    }

    /* Record summary. */
    oskar_log_section(h->log, 'M', "Imaging complete");
    if (!*status)
    {
        if (h->num_vis_processed > 1e6)
        {
            oskar_log_value(h->log, 'M', 0,
                    "Visibilities processed", "%.3f million",
                    h->num_vis_processed * 1e-6);
        }
        else
        {
            oskar_log_value(h->log, 'M', 0,
                    "Visibilities processed", "%lu",
                    (unsigned long) (h->num_vis_processed));
        }
        if (h->num_w_planes > 0)
        {
            oskar_log_value(h->log, 'M', 0,
                    "W-projection planes", "%d", h->num_w_planes);
        }
        if (h->fov_deg > 0.1)
        {
            oskar_log_value(h->log, 'M', 0,
                    "Field of view [deg]", "%.1f", h->fov_deg);
        }
        else
        {
            oskar_log_value(h->log, 'M', 0,
                    "Field of view [arcmin]", "%.1f", h->fov_deg * 60.0);
        }
        oskar_log_value(h->log, 'M', 0,
                "Image dimension [pixels]", "%d", h->image_size);
        if (h->num_files > 0)
        {
            oskar_log_message(h->log, 'M', 0, "Input(s):");
            for (i = 0; i < h->num_files; ++i)
            {
                oskar_log_value(h->log, 'M', 1, "Visibility data", "%s",
                        h->input_files[i]);
            }
        }
        if (h->output_root)
        {
            oskar_log_message(h->log, 'M', 0, "Output(s):");
            for (i = 0; i < h->num_im_pols; ++i)
            {
                oskar_log_value(h->log, 'M', 1, "FITS file", "%s (%.1f MB)",
                        h->output_name[i],
                        num_pix * h->num_im_channels *
                        oskar_mem_element_size(h->imager_prec) / 1e6);
            }
        }
        oskar_log_message(h->log, 'M', 0, "Run completed in %.3f sec.",
                oskar_timer_elapsed(h->tmr_overall));
    }
    else
    {
        oskar_log_error(h->log, "Run failed with code %i: %s.", *status,
                oskar_get_error_string(*status));
    }

    /* Write log to the output FITS files as HISTORY entries.
     * Replace newlines with zeros. */
    log_data = oskar_log_file_data(h->log, &log_size);
    for (j = 0; j < log_size; ++j)
    {
        if (log_data[j] == '\n') log_data[j] = 0;
        if (log_data[j] == '\r') log_data[j] = ' ';
    }
    for (i = 0; i < h->num_im_pols; ++i)
    {
        const char* line = log_data;
        if (!h->fits_file[i]) continue;
        length = log_size;
        for (; log_size > 0;)
        {
            const char* eol = 0;
            fits_write_history(h->fits_file[i], line, status);
            eol = (const char*) memchr(line, '\0', length);
            if (!eol) break;
            eol += 1;
            length -= (eol - line);
            line = eol;
        }
    }
    free(log_data);

    /* Reset imager memory. */
    oskar_imager_reset_cache(h, status);

    /* Close the log. */
    oskar_log_close(h->log);
}


void oskar_imager_finalise_plane(oskar_Imager* h,
        oskar_Mem* plane, double plane_norm, int* status)
{
    if (*status) return;

    /* Apply normalisation. */
    if (plane_norm > 0.0 || plane_norm < 0.0)
    {
        oskar_timer_resume(h->tmr_grid_finalise);
        oskar_mem_scale_real(plane, 1.0 / plane_norm,
                0, oskar_mem_length(plane), status);
        oskar_timer_pause(h->tmr_grid_finalise);
    }

    /* If algorithm if DFT, we've finished here. */
    if (h->algorithm == OSKAR_ALGORITHM_DFT_2D ||
            h->algorithm == OSKAR_ALGORITHM_DFT_3D)
    {
        return;
    }

    /* Check plane is complex type, as plane must be gridded visibilities. */
    if (!oskar_mem_is_complex(plane))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check plane size is as expected. */
    const int size = oskar_imager_plane_size(h);
    if (oskar_mem_length(plane) != ((size_t)size * (size_t)size))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Perform FFT shift of the input grid. */
    oskar_timer_resume(h->tmr_grid_finalise);
    const int fft_loc = (h->fft_on_gpu && h->num_gpus > 0) ?
            h->dev_loc : OSKAR_CPU;
    if (fft_loc != OSKAR_CPU)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[0], status);
    }
    oskar_fftphase(size, size, plane, status);

    /* Call FFT. */
    if (!h->fft)
    {
        h->fft = oskar_fft_create(h->imager_prec, fft_loc, 2, size, 0, status);
    }
    oskar_fft_exec(h->fft, plane, status);

    /* Generate grid correction function if required. */
    if (!h->corr_func)
    {
        oskar_Mem* corr_func = 0;
        corr_func = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, size, status);
        if (h->algorithm != OSKAR_ALGORITHM_FFT)
        {
            oskar_grid_correction_function_spheroidal(size, h->oversample,
                    oskar_mem_double(corr_func, status));
        }
        else
        {
            if (h->kernel_type == 'S')
            {
                oskar_grid_correction_function_spheroidal(size, 0,
                        oskar_mem_double(corr_func, status));
            }
            else if (h->kernel_type == 'P')
            {
                oskar_grid_correction_function_pillbox(size,
                        oskar_mem_double(corr_func, status));
            }
        }
        h->corr_func = oskar_mem_convert_precision(corr_func,
                h->imager_prec, status);
    }

    /* FFT shift again, and apply grid correction. */
    oskar_fftphase(size, size, plane, status);
    oskar_grid_correction(size, h->corr_func, plane, status);
    oskar_timer_pause(h->tmr_grid_finalise);
}


void oskar_imager_trim_image(oskar_Imager* h, oskar_Mem* plane,
        int plane_size, int image_size, int* status)
{
    if (*status) return;

    /* Get the real part only, if the plane is complex. */
    oskar_timer_resume(h->tmr_grid_finalise);
    if (oskar_mem_is_complex(plane))
    {
        size_t i = 0;
        const size_t num_cells = (size_t)plane_size * (size_t)plane_size;
        if (oskar_mem_precision(plane) == OSKAR_DOUBLE)
        {
            double *t = oskar_mem_double(plane, status);
            for (i = 0; i < num_cells; ++i) t[i] = t[2 * i];
        }
        else
        {
            float *t = oskar_mem_float(plane, status);
            for (i = 0; i < num_cells; ++i) t[i] = t[2 * i];
        }
    }

    /* Trim to required image size. */
    const int size_diff = plane_size - image_size;
    if (size_diff > 0)
    {
        char *ptr = 0;
        size_t in = 0, out = 0, element_size = 0;
        int i = 0;
        ptr = oskar_mem_char(plane);
        element_size = oskar_mem_element_size(oskar_mem_precision(plane));
        in = element_size * (size_diff / 2) * (plane_size + 1);
        const size_t copy_len = element_size * image_size;
        for (i = 0; i < image_size; ++i)
        {
            /* Use memmove() instead of memcpy() to allow for overlap. */
            memmove(ptr + out, ptr + in, copy_len);
            in += plane_size * element_size;
            out += copy_len;
        }
    }
    oskar_timer_pause(h->tmr_grid_finalise);
}


void write_plane(oskar_Imager* h, oskar_Mem* plane,
        int c, int p, int* status)
{
    long firstpix[3];
    if (*status) return;
    if (!h->fits_file[p]) return;
    const int datatype = (oskar_mem_is_double(plane) ? TDOUBLE : TFLOAT);
    firstpix[0] = 1;
    firstpix[1] = 1;
    firstpix[2] = 1 + c;
    const int num_pixels = h->image_size * h->image_size;
    fits_write_pix(h->fits_file[p], datatype, firstpix, num_pixels,
            oskar_mem_void(plane), status);
}


#ifdef __cplusplus
}
#endif
