/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "splines/oskar_dierckx_bispev.h"
#include "splines/oskar_splines.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_splines_evaluate(const oskar_Splines* spline,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        int stride_out, int offset_out, oskar_Mem* output, int* status)
{
    if (*status) return;
    const int type = oskar_splines_precision(spline);
    const int location = oskar_splines_mem_location(spline);
    const int nx = oskar_splines_num_knots_x_theta(spline);
    const int ny = oskar_splines_num_knots_y_phi(spline);
    const oskar_Mem* tx = oskar_splines_knots_x_theta_const(spline);
    const oskar_Mem* ty = oskar_splines_knots_y_phi_const(spline);
    const oskar_Mem* coeff = oskar_splines_coeff_const(spline);
    if (type != oskar_mem_type(x) || type != oskar_mem_type(y))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(output) ||
            location != oskar_mem_location(x) ||
            location != oskar_mem_location(y))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        int i = 0;
        if (type == OSKAR_SINGLE)
        {
            const float *tx_ = 0, *ty_ = 0, *coeff_ = 0, *x_ = 0, *y_ = 0;
            float *out = 0;
            tx_    = oskar_mem_float_const(tx, status);
            ty_    = oskar_mem_float_const(ty, status);
            coeff_ = oskar_mem_float_const(coeff, status);
            x_     = oskar_mem_float_const(x, status);
            y_     = oskar_mem_float_const(y, status);
            out    = oskar_mem_float(output, status) + offset_out;
            if (nx == 0 || ny == 0 || !tx_ || !ty_ || !coeff_)
            {
                for (i = 0; i < num_points; ++i) out[i * stride_out] = 0.0f;
            }
            else
            {
                float x1 = 0.0, y1 = 0.0, wrk[8];
                int iwrk1[2], kwrk1 = 2, lwrk = 8, err = 0;
                for (i = 0; i < num_points; ++i)
                {
                    x1 = x_[i];
                    y1 = y_[i];
                    oskar_dierckx_bispev_f(tx_, nx, ty_, ny, coeff_, 3, 3,
                            &x1, 1, &y1, 1, &out[i * stride_out],
                            wrk, lwrk, iwrk1, kwrk1, &err);
                    if (err != 0)
                    {
                        *status = OSKAR_ERR_SPLINE_EVAL_FAIL;
                        return;
                    }
                }
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            const double *tx_ = 0, *ty_ = 0, *coeff_ = 0, *x_ = 0, *y_ = 0;
            double *out = 0;
            tx_    = oskar_mem_double_const(tx, status);
            ty_    = oskar_mem_double_const(ty, status);
            coeff_ = oskar_mem_double_const(coeff, status);
            x_     = oskar_mem_double_const(x, status);
            y_     = oskar_mem_double_const(y, status);
            out    = oskar_mem_double(output, status) + offset_out;
            if (nx == 0 || ny == 0 || !tx_ || !ty_ || !coeff_)
            {
                for (i = 0; i < num_points; ++i) out[i * stride_out] = 0.0;
            }
            else
            {
                double x1 = 0.0, y1 = 0.0, wrk[8];
                int iwrk1[2], kwrk1 = 2, lwrk = 8, err = 0;
                for (i = 0; i < num_points; ++i)
                {
                    x1 = x_[i];
                    y1 = y_[i];
                    oskar_dierckx_bispev_d(tx_, nx, ty_, ny, coeff_, 3, 3,
                            &x1, 1, &y1, 1, &out[i * stride_out],
                            wrk, lwrk, iwrk1, kwrk1, &err);
                    if (err != 0)
                    {
                        *status = OSKAR_ERR_SPLINE_EVAL_FAIL;
                        return;
                    }
                }
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (nx == 0 || ny == 0)
        {
            if (type == OSKAR_DOUBLE)
            {
                k = "set_zeros_stride_double";
            }
            else if (type == OSKAR_SINGLE)
            {
                k = "set_zeros_stride_float";
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(
                    (size_t) num_points, local_size[0]);
            const oskar_Arg args[] = {
                    {INT_SZ, &num_points},
                    {INT_SZ, &stride_out},
                    {INT_SZ, &offset_out},
                    {PTR_SZ, oskar_mem_buffer(output)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
        else
        {
            if (type == OSKAR_DOUBLE)
            {
                k = "dierckx_bispev_bicubic_double";
            }
            else if (type == OSKAR_SINGLE)
            {
                k = "dierckx_bispev_bicubic_float";
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(
                    (size_t) num_points, local_size[0]);
            const oskar_Arg args[] = {
                    {PTR_SZ, oskar_mem_buffer_const(tx)},
                    {INT_SZ, &nx},
                    {PTR_SZ, oskar_mem_buffer_const(ty)},
                    {INT_SZ, &ny},
                    {PTR_SZ, oskar_mem_buffer_const(coeff)},
                    {INT_SZ, &num_points},
                    {PTR_SZ, oskar_mem_buffer_const(x)},
                    {PTR_SZ, oskar_mem_buffer_const(y)},
                    {INT_SZ, &stride_out},
                    {INT_SZ, &offset_out},
                    {PTR_SZ, oskar_mem_buffer(output)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
