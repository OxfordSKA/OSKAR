/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_dftw_c2c.h"
#include "math/define_dftw_m2m.h"
#include "math/define_multiply.h"
#include "math/oskar_dftw.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

#define DBL (1 << 0)
#define FLT (0 << 0)
#define D3  (1 << 1)
#define D2  (0 << 1)
#define MAT (1 << 2)

OSKAR_DFTW_C2C_CPU(dftw_c2c_2d_float, 0, float, float2)
OSKAR_DFTW_C2C_CPU(dftw_c2c_3d_float, 1, float, float2)
OSKAR_DFTW_M2M_CPU(dftw_m2m_2d_float, 0, float, float2)
OSKAR_DFTW_M2M_CPU(dftw_m2m_3d_float, 1, float, float2)

OSKAR_DFTW_C2C_CPU(dftw_c2c_2d_double, 0, double, double2)
OSKAR_DFTW_C2C_CPU(dftw_c2c_3d_double, 1, double, double2)
OSKAR_DFTW_M2M_CPU(dftw_m2m_2d_double, 0, double, double2)
OSKAR_DFTW_M2M_CPU(dftw_m2m_3d_double, 1, double, double2)

static int get_block_size(int num_total)
{
    const int warp_size = 32;
    const int num_warps = (num_total + warp_size - 1) / warp_size;
    const int block_size = num_warps * warp_size;
    return ((block_size > 256) ? 256 : block_size);
}

void oskar_dftw(
        int normalise,
        int num_in,
        double wavenumber,
        const oskar_Mem* weights_in,
        const oskar_Mem* x_in,
        const oskar_Mem* y_in,
        const oskar_Mem* z_in,
        int offset_coord_out,
        int num_out,
        const oskar_Mem* x_out,
        const oskar_Mem* y_out,
        const oskar_Mem* z_out,
        const oskar_Mem* data_idx,
        const oskar_Mem* data,
        int eval_x,
        int eval_y,
        int offset_out,
        oskar_Mem* output,
        int* status)
{
    double norm_factor = 0.0;
    float norm_factor_f = 0.0f;
    if (*status) return;
    const int location = oskar_mem_location(output);
    const int type = oskar_mem_precision(output);
    const int is_dbl = oskar_mem_is_double(output);
    const int is_3d = (z_in != NULL && z_out != NULL);
    const int is_matrix = oskar_mem_is_matrix(output);
    norm_factor = normalise ? 1.0 / num_in : 1.0;
    norm_factor_f = (float) norm_factor;
    if (!oskar_mem_is_complex(output) || !oskar_mem_is_complex(weights_in) ||
            oskar_mem_is_matrix(weights_in))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_location(data) != location ||
            oskar_mem_location(weights_in) != location ||
            oskar_mem_location(x_in) != location ||
            oskar_mem_location(y_in) != location ||
            oskar_mem_location(x_out) != location ||
            oskar_mem_location(y_out) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_precision(weights_in) != type ||
            oskar_mem_type(x_in) != type ||
            oskar_mem_type(y_in) != type ||
            oskar_mem_type(x_out) != type ||
            oskar_mem_type(y_out) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (!oskar_mem_is_complex(data) ||
            oskar_mem_type(data) != oskar_mem_type(output) ||
            oskar_mem_precision(data) != type)
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
    oskar_mem_ensure(output, (size_t) offset_out + num_out, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        const int* data_idx_p =
                data_idx ? oskar_mem_int_const(data_idx, status) : 0;
        if (is_matrix)
        {
            if (is_3d)
            {
                if (is_dbl)
                {
                    dftw_m2m_3d_double(num_in, wavenumber,
                            oskar_mem_double2_const(weights_in, status),
                            oskar_mem_double_const(x_in, status),
                            oskar_mem_double_const(y_in, status),
                            oskar_mem_double_const(z_in, status),
                            offset_coord_out, num_out,
                            oskar_mem_double_const(x_out, status),
                            oskar_mem_double_const(y_out, status),
                            oskar_mem_double_const(z_out, status),
                            data_idx_p,
                            oskar_mem_double2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_double2(output, status), norm_factor, 0);
                }
                else
                {
                    dftw_m2m_3d_float(num_in, (float)wavenumber,
                            oskar_mem_float2_const(weights_in, status),
                            oskar_mem_float_const(x_in, status),
                            oskar_mem_float_const(y_in, status),
                            oskar_mem_float_const(z_in, status),
                            offset_coord_out, num_out,
                            oskar_mem_float_const(x_out, status),
                            oskar_mem_float_const(y_out, status),
                            oskar_mem_float_const(z_out, status),
                            data_idx_p,
                            oskar_mem_float2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_float2(output, status), norm_factor_f, 0);
                }
            }
            else
            {
                if (is_dbl)
                {
                    dftw_m2m_2d_double(num_in, wavenumber,
                            oskar_mem_double2_const(weights_in, status),
                            oskar_mem_double_const(x_in, status),
                            oskar_mem_double_const(y_in, status), 0,
                            offset_coord_out, num_out,
                            oskar_mem_double_const(x_out, status),
                            oskar_mem_double_const(y_out, status), 0,
                            data_idx_p,
                            oskar_mem_double2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_double2(output, status), norm_factor, 0);
                }
                else
                {
                    dftw_m2m_2d_float(num_in, (float)wavenumber,
                            oskar_mem_float2_const(weights_in, status),
                            oskar_mem_float_const(x_in, status),
                            oskar_mem_float_const(y_in, status), 0,
                            offset_coord_out, num_out,
                            oskar_mem_float_const(x_out, status),
                            oskar_mem_float_const(y_out, status), 0,
                            data_idx_p,
                            oskar_mem_float2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_float2(output, status), norm_factor_f, 0);
                }
            }
        }
        else
        {
            if (is_3d)
            {
                if (is_dbl)
                {
                    dftw_c2c_3d_double(num_in, wavenumber,
                            oskar_mem_double2_const(weights_in, status),
                            oskar_mem_double_const(x_in, status),
                            oskar_mem_double_const(y_in, status),
                            oskar_mem_double_const(z_in, status),
                            offset_coord_out, num_out,
                            oskar_mem_double_const(x_out, status),
                            oskar_mem_double_const(y_out, status),
                            oskar_mem_double_const(z_out, status),
                            data_idx_p,
                            oskar_mem_double2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_double2(output, status), norm_factor, 0);
                }
                else
                {
                    dftw_c2c_3d_float(num_in, (float)wavenumber,
                            oskar_mem_float2_const(weights_in, status),
                            oskar_mem_float_const(x_in, status),
                            oskar_mem_float_const(y_in, status),
                            oskar_mem_float_const(z_in, status),
                            offset_coord_out, num_out,
                            oskar_mem_float_const(x_out, status),
                            oskar_mem_float_const(y_out, status),
                            oskar_mem_float_const(z_out, status),
                            data_idx_p,
                            oskar_mem_float2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_float2(output, status), norm_factor_f, 0);
                }
            }
            else
            {
                if (is_dbl)
                {
                    dftw_c2c_2d_double(num_in, wavenumber,
                            oskar_mem_double2_const(weights_in, status),
                            oskar_mem_double_const(x_in, status),
                            oskar_mem_double_const(y_in, status), 0,
                            offset_coord_out, num_out,
                            oskar_mem_double_const(x_out, status),
                            oskar_mem_double_const(y_out, status), 0,
                            data_idx_p,
                            oskar_mem_double2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_double2(output, status), norm_factor, 0);
                }
                else
                {
                    dftw_c2c_2d_float(num_in, (float)wavenumber,
                            oskar_mem_float2_const(weights_in, status),
                            oskar_mem_float_const(x_in, status),
                            oskar_mem_float_const(y_in, status), 0,
                            offset_coord_out, num_out,
                            oskar_mem_float_const(x_out, status),
                            oskar_mem_float_const(y_out, status), 0,
                            data_idx_p,
                            oskar_mem_float2_const(data, status),
                            eval_x, eval_y, offset_out,
                            oskar_mem_float2(output, status), norm_factor_f, 0);
                }
            }
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const void* np = 0;
        const char* k = 0;
        int max_in_chunk = 0;
        float wavenumber_f = (float) wavenumber;

        /* Select the kernel. */
        switch (is_dbl * DBL | is_3d * D3 | is_matrix * MAT)
        {
        case D2 | FLT:       k = "dftw_c2c_2d_float";  break;
        case D2 | DBL:       k = "dftw_c2c_2d_double"; break;
        case D3 | FLT:       k = "dftw_c2c_3d_float";  break;
        case D3 | DBL:       k = "dftw_c2c_3d_double"; break;
        case D2 | FLT | MAT: k = "dftw_m2m_2d_float";  break;
        case D2 | DBL | MAT: k = "dftw_m2m_2d_double"; break;
        case D3 | FLT | MAT: k = "dftw_m2m_3d_float";  break;
        case D3 | DBL | MAT: k = "dftw_m2m_3d_double"; break;
        default:
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }
        if (oskar_device_is_nv(location))
        {
            local_size[0] = (size_t) get_block_size(num_out);
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_out, local_size[0]);

        /* max_in_chunk must be multiple of 16. */
        max_in_chunk = 256;
        const size_t fp_size = is_dbl ? sizeof(double) : sizeof(float);
        const size_t arg_size_local[] = {
                2 * (max_in_chunk * fp_size),
                2 * (max_in_chunk * fp_size),
                (is_3d ? (max_in_chunk * fp_size) : 0),
                (max_in_chunk * sizeof(int))
        };

        /* Set kernel arguments. */
        const oskar_Arg args[] = {
                {INT_SZ, &num_in},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (void*)&wavenumber : (void*)&wavenumber_f},
                {PTR_SZ, oskar_mem_buffer_const(weights_in)},
                {PTR_SZ, oskar_mem_buffer_const(x_in)},
                {PTR_SZ, oskar_mem_buffer_const(y_in)},
                {PTR_SZ, is_3d ? oskar_mem_buffer_const(z_in) : &np},
                {INT_SZ, &offset_coord_out},
                {INT_SZ, &num_out},
                {PTR_SZ, oskar_mem_buffer_const(x_out)},
                {PTR_SZ, oskar_mem_buffer_const(y_out)},
                {PTR_SZ, is_3d ? oskar_mem_buffer_const(z_out) : &np},
                {PTR_SZ, data_idx ? oskar_mem_buffer_const(data_idx) : &np},
                {PTR_SZ, oskar_mem_buffer_const(data)},
                {INT_SZ, &eval_x},
                {INT_SZ, &eval_y},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(output)},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (void*)&norm_factor : (void*)&norm_factor_f},
                {INT_SZ, &max_in_chunk}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args,
                sizeof(arg_size_local) / sizeof(size_t), arg_size_local,
                status);
    }
}
