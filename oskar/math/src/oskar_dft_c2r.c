/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_dft_c2r.h"
#include "math/oskar_dft_c2r.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

OSKAR_DFT_C2R_CPU(dft_c2r_2d_float, 0, float, float2)
OSKAR_DFT_C2R_CPU(dft_c2r_3d_float, 1, float, float2)
OSKAR_DFT_C2R_CPU(dft_c2r_2d_double, 0, double, double2)
OSKAR_DFT_C2R_CPU(dft_c2r_3d_double, 1, double, double2)

static int oskar_int_range_clamp(int value, int minimum, int maximum)
{
   if (value < minimum) return minimum;
   if (value > maximum) return maximum;
   return value;
}

void oskar_dft_c2r(
        int num_in,
        double wavenumber,
        const oskar_Mem* x_in,
        const oskar_Mem* y_in,
        const oskar_Mem* z_in,
        const oskar_Mem* data_in,
        const oskar_Mem* weights_in,
        int num_out,
        const oskar_Mem* x_out,
        const oskar_Mem* y_out,
        const oskar_Mem* z_out,
        oskar_Mem* output,
        int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(output);
    const int type = oskar_mem_precision(output);
    const int is_dbl = oskar_mem_is_double(output);
    const int is_3d =
            (z_in != NULL && z_out != NULL && oskar_mem_length(z_out) > 0);
    if (!oskar_mem_is_complex(data_in) ||
            oskar_mem_is_complex(output) ||
            oskar_mem_is_complex(weights_in) ||
            oskar_mem_is_matrix(weights_in))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_location(weights_in) != location ||
            oskar_mem_location(x_in) != location ||
            oskar_mem_location(y_in) != location ||
            oskar_mem_location(x_out) != location ||
            oskar_mem_location(y_out) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_precision(data_in) != type ||
            oskar_mem_precision(weights_in) != type ||
            oskar_mem_type(x_in) != type ||
            oskar_mem_type(y_in) != type ||
            oskar_mem_type(x_out) != type ||
            oskar_mem_type(y_out) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (is_3d)
    {
        if (oskar_mem_location(z_in) != location ||
                oskar_mem_location(z_out) != location)
        {
            *status = OSKAR_ERR_LOCATION_MISMATCH;
            return;
        }
        if (oskar_mem_type(z_in) != type || oskar_mem_type(z_out) != type)
        {
            *status = OSKAR_ERR_TYPE_MISMATCH;
            return;
        }
    }
    oskar_mem_ensure(output, (size_t) num_out, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        if (is_3d)
        {
            if (is_dbl)
            {
                dft_c2r_3d_double(num_in, wavenumber,
                        oskar_mem_double_const(x_in, status),
                        oskar_mem_double_const(y_in, status),
                        oskar_mem_double_const(z_in, status),
                        oskar_mem_double2_const(data_in, status),
                        oskar_mem_double_const(weights_in, status), 0, num_out,
                        oskar_mem_double_const(x_out, status),
                        oskar_mem_double_const(y_out, status),
                        oskar_mem_double_const(z_out, status), 0,
                        oskar_mem_double(output, status), 0);
            }
            else
            {
                dft_c2r_3d_float(num_in, (float)wavenumber,
                        oskar_mem_float_const(x_in, status),
                        oskar_mem_float_const(y_in, status),
                        oskar_mem_float_const(z_in, status),
                        oskar_mem_float2_const(data_in, status),
                        oskar_mem_float_const(weights_in, status), 0, num_out,
                        oskar_mem_float_const(x_out, status),
                        oskar_mem_float_const(y_out, status),
                        oskar_mem_float_const(z_out, status), 0,
                        oskar_mem_float(output, status), 0);
            }
        }
        else
        {
            if (is_dbl)
            {
                dft_c2r_2d_double(num_in, wavenumber,
                        oskar_mem_double_const(x_in, status),
                        oskar_mem_double_const(y_in, status), 0,
                        oskar_mem_double2_const(data_in, status),
                        oskar_mem_double_const(weights_in, status), 0, num_out,
                        oskar_mem_double_const(x_out, status),
                        oskar_mem_double_const(y_out, status), 0, 0,
                        oskar_mem_double(output, status), 0);
            }
            else
            {
                dft_c2r_2d_float(num_in, (float)wavenumber,
                        oskar_mem_float_const(x_in, status),
                        oskar_mem_float_const(y_in, status), 0,
                        oskar_mem_float2_const(data_in, status),
                        oskar_mem_float_const(weights_in, status), 0, num_out,
                        oskar_mem_float_const(x_out, status),
                        oskar_mem_float_const(y_out, status), 0, 0,
                        oskar_mem_float(output, status), 0);
            }
        }
    }
    else
    {
        size_t local_size[] = {1, 1, 1}, global_size[] = {1, 1, 1};
        float wavenumber_f = (float) wavenumber;
        const void* np = 0;
        const char* k = 0;
        int out_size = 0, max_out_size = 0, start = 0;
        local_size[0] = oskar_device_is_nv(location) ? 384 : 256;
        oskar_device_check_local_size(location, 0, local_size);

        /* Select the kernel. */
        if (is_3d)
        {
            k = is_dbl ? "dft_c2r_3d_double" : "dft_c2r_3d_float";
        }
        else
        {
            k = is_dbl ? "dft_c2r_2d_double" : "dft_c2r_2d_float";
        }

        /* Compute the maximum manageable output chunk size. */
        /* Product of max output and input sizes. */
        max_out_size = 8192 * (is_dbl ? 32768 : 65536);
        max_out_size /= num_in;
        max_out_size = (int) oskar_device_global_size(
                (size_t) max_out_size, local_size[0]); /* Last was 1024. */
        max_out_size = oskar_int_range_clamp(max_out_size,
                (int) local_size[0] * 2,
                (int) local_size[0] * (is_dbl ? 80 : 160));
        /* max_in_chunk must be multiple of 16. */
        const int max_in_chunk = is_3d ?
                (is_dbl ? 384 : 800) : (is_dbl ? 448 : 896);
        const size_t element_size = is_dbl ? sizeof(double) : sizeof(float);
        const size_t local_mem_size = max_in_chunk * element_size;
        const size_t arg_size_local[] = {
                2 * local_mem_size, 2 * local_mem_size,
                (is_3d ? local_mem_size : 0)
        };

        /* Loop over output chunks. */
        for (start = 0; start < num_out; start += max_out_size)
        {
            if (*status) break;

            /* Get the chunk size. */
            out_size = num_out - start;
            if (out_size > max_out_size) out_size = max_out_size;
            global_size[0] = oskar_device_global_size(out_size, local_size[0]);

            /* Set kernel arguments. */
            const oskar_Arg args[] = {
                    {INT_SZ, &num_in},
                    {is_dbl ? DBL_SZ : FLT_SZ,
                            is_dbl ? (void*)&wavenumber : (void*)&wavenumber_f},
                    {PTR_SZ, oskar_mem_buffer_const(x_in)},
                    {PTR_SZ, oskar_mem_buffer_const(y_in)},
                    {PTR_SZ, is_3d ? oskar_mem_buffer_const(z_in) : &np},
                    {PTR_SZ, oskar_mem_buffer_const(data_in)},
                    {PTR_SZ, oskar_mem_buffer_const(weights_in)},
                    {INT_SZ, &start},
                    {INT_SZ, &out_size},
                    {PTR_SZ, oskar_mem_buffer_const(x_out)},
                    {PTR_SZ, oskar_mem_buffer_const(y_out)},
                    {PTR_SZ, is_3d ? oskar_mem_buffer_const(z_out) : &np},
                    {INT_SZ, &start},
                    {PTR_SZ, oskar_mem_buffer(output)},
                    {INT_SZ, &max_in_chunk}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args,
                    sizeof(arg_size_local) / sizeof(size_t), arg_size_local,
                    status);
        }
    }
}
