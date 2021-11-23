/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/oskar_imager.h"

#include "imager/private_imager_select_data.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C0 299792458.0

static
void copy_vis_pol(size_t num_rows, int num_channels, int num_pols,
        int c, int p, const oskar_Mem* vis_in, const oskar_Mem* weight_in,
        oskar_Mem* vis_out, oskar_Mem* weight_out, size_t out_offset,
        int* status);

#define COPY_COORDS_CPU(FP, SCALE) {\
    size_t r = 0;\
    FP *uu_o = 0, *vv_o = 0, *ww_o = 0;\
    const FP *uu_i = 0, *vv_i = 0, *ww_i = 0;\
    uu_o = ((FP*) oskar_mem_void(uu_out)) + *num_out;\
    vv_o = ((FP*) oskar_mem_void(vv_out)) + *num_out;\
    ww_o = ((FP*) oskar_mem_void(ww_out)) + *num_out;\
    uu_i = ((const FP*) oskar_mem_void_const(uu_in));\
    vv_i = ((const FP*) oskar_mem_void_const(vv_in));\
    ww_i = ((const FP*) oskar_mem_void_const(ww_in));\
    for (r = 0; r < num_rows; ++r) {\
        uu_o[r] = uu_i[r] * SCALE;\
        vv_o[r] = vv_i[r] * SCALE;\
        ww_o[r] = ww_i[r] * SCALE;\
    }\
    }


void oskar_imager_select_data(
        const oskar_Imager* h,
        size_t num_rows,
        int start_chan,
        int end_chan,
        int num_pols,
        const oskar_Mem* uu_in,
        const oskar_Mem* vv_in,
        const oskar_Mem* ww_in,
        const oskar_Mem* vis_in,
        const oskar_Mem* weight_in,
        const oskar_Mem* time_in,
        double im_freq_hz,
        int im_pol,
        size_t* num_out,
        oskar_Mem* uu_out,
        oskar_Mem* vv_out,
        oskar_Mem* ww_out,
        oskar_Mem* vis_out,
        oskar_Mem* weight_out,
        oskar_Mem* time_out,
        int* status)
{
    int i = 0, c = 0, p = 0;
    const double s = 0.05;
    const double df = h->freq_inc_hz != 0.0 ? h->freq_inc_hz : 1.0;
    const double f0 = h->vis_freq_start_hz;
    const int location = oskar_mem_location(uu_in);
    const int prec = oskar_mem_type(uu_in);

    /* Initialise. */
    if (*status) return;
    *num_out = 0;

    /* Override pol_offset if required. */
    p = h->pol_offset;
    if (h->im_type == OSKAR_IMAGE_TYPE_STOKES ||
            h->im_type == OSKAR_IMAGE_TYPE_LINEAR)
    {
        p = im_pol;
    }
    if (num_pols == 1) p = 0;

    /* Check whether using frequency snapshots or frequency synthesis. */
    const int num_channels = 1 + end_chan - start_chan;
    if (h->chan_snaps)
    {
        /* Get the channel for the image and check if out of range. */
        c = (int) round((im_freq_hz - f0) / df);
        if (c < start_chan || c > end_chan) return;
        if (fabs((im_freq_hz - f0) - c * df) > s * df) return;
        const double inv_wavelength = (f0 + c * df) / C0;

        /* Copy the baseline coordinates in wavelengths. */
        if (location == OSKAR_CPU)
        {
            if (prec == OSKAR_SINGLE)
            {
                const float inv_wavelength_f = (float) inv_wavelength;
                COPY_COORDS_CPU(float, inv_wavelength_f)
            }
            else
            {
                COPY_COORDS_CPU(double, inv_wavelength)
            }
        }
        else
        {
            oskar_mem_copy_contents(uu_out, uu_in, 0, 0, num_rows, status);
            oskar_mem_copy_contents(vv_out, vv_in, 0, 0, num_rows, status);
            oskar_mem_copy_contents(ww_out, ww_in, 0, 0, num_rows, status);
            oskar_mem_scale_real(uu_out, inv_wavelength, 0, num_rows, status);
            oskar_mem_scale_real(vv_out, inv_wavelength, 0, num_rows, status);
            oskar_mem_scale_real(ww_out, inv_wavelength, 0, num_rows, status);
        }

        /* Copy visibility data and weights if present. */
        copy_vis_pol(num_rows, num_channels, num_pols, c - start_chan, p,
                (h->coords_only ? 0 : vis_in), weight_in,
                (h->coords_only ? 0 : vis_out), weight_out,
                0, status);

        /* Copy time centroids if present. */
        if (time_in && time_out)
        {
            oskar_mem_ensure(time_out, num_rows, status);
            oskar_mem_copy_contents(time_out, time_in, 0, 0, num_rows, status);
        }
        *num_out += num_rows;
    }
    else /* Frequency synthesis */
    {
        for (i = 0; i < h->num_sel_freqs; ++i)
        {
            c = (int) round((h->sel_freqs[i] - f0) / df);
            if (c < start_chan || c > end_chan) continue;
            if (fabs((h->sel_freqs[i] - f0) - c * df) > s * df) continue;
            const double inv_wavelength = (f0 + c * df) / C0;

            /* Copy the baseline coordinates in wavelengths. */
            if (location == OSKAR_CPU)
            {
                if (prec == OSKAR_SINGLE)
                {
                    const float inv_wavelength_f = (float) inv_wavelength;
                    COPY_COORDS_CPU(float, inv_wavelength_f)
                }
                else
                {
                    COPY_COORDS_CPU(double, inv_wavelength)
                }
            }
            else
            {
                oskar_mem_copy_contents(uu_out, uu_in,
                        *num_out, 0, num_rows, status);
                oskar_mem_copy_contents(vv_out, vv_in,
                        *num_out, 0, num_rows, status);
                oskar_mem_copy_contents(ww_out, ww_in,
                        *num_out, 0, num_rows, status);
                oskar_mem_scale_real(uu_out, inv_wavelength,
                        *num_out, num_rows, status);
                oskar_mem_scale_real(vv_out, inv_wavelength,
                        *num_out, num_rows, status);
                oskar_mem_scale_real(ww_out, inv_wavelength,
                        *num_out, num_rows, status);
            }

            /* Copy visibility data and weights if present. */
            copy_vis_pol(num_rows, num_channels, num_pols, c - start_chan, p,
                    (h->coords_only ? 0 : vis_in), weight_in,
                    (h->coords_only ? 0 : vis_out), weight_out,
                    *num_out, status);

            /* Copy time centroids if present. */
            if (time_in && time_out)
            {
                oskar_mem_ensure(time_out, num_rows * num_channels, status);
                oskar_mem_copy_contents(time_out, time_in, *num_out, 0,
                        num_rows, status);
            }
            *num_out += num_rows;
        }
    }
}


void copy_vis_pol(size_t num_rows, int num_channels, int num_pols,
        int c, int p, const oskar_Mem* vis_in, const oskar_Mem* weight_in,
        oskar_Mem* vis_out, oskar_Mem* weight_out, size_t out_offset,
        int* status)
{
    size_t r = 0;
    if (*status) return;
    if (num_pols == 1 && num_channels == 1)
    {
        oskar_mem_copy_contents(weight_out, weight_in,
                out_offset, 0, num_rows, status);
        if (vis_in && vis_out)
        {
            oskar_mem_copy_contents(vis_out, vis_in,
                    out_offset, 0, num_rows, status);
        }
    }
    else
    {
        if (oskar_mem_precision(weight_out) == OSKAR_SINGLE)
        {
            float* w_out = 0;
            const float* w_in = 0;
            w_out = oskar_mem_float(weight_out, status) + out_offset;
            w_in = oskar_mem_float_const(weight_in, status);
            for (r = 0; r < num_rows; ++r)
            {
                w_out[r] = w_in[num_pols * r + p];
            }

            if (vis_in && vis_out)
            {
                float2* v_out = 0;
                const float2* v_in = 0;
                v_out = oskar_mem_float2(vis_out, status) + out_offset;
                v_in = oskar_mem_float2_const(vis_in, status);
                for (r = 0; r < num_rows; ++r)
                {
                    v_out[r] = v_in[num_pols * (num_channels * r + c) + p];
                }
            }
        }
        else
        {
            double* w_out = 0;
            const double* w_in = 0;
            w_out = oskar_mem_double(weight_out, status) + out_offset;
            w_in = oskar_mem_double_const(weight_in, status);
            for (r = 0; r < num_rows; ++r)
            {
                w_out[r] = w_in[num_pols * r + p];
            }

            if (vis_in && vis_out)
            {
                double2* v_out = 0;
                const double2* v_in = 0;
                v_out = oskar_mem_double2(vis_out, status) + out_offset;
                v_in = oskar_mem_double2_const(vis_in, status);
                for (r = 0; r < num_rows; ++r)
                {
                    v_out[r] = v_in[num_pols * (num_channels * r + c) + p];
                }
            }
        }
    }
}


#ifdef __cplusplus
}
#endif
