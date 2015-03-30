/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_evaluate_array_pattern.h>
#include <oskar_dftw_o2c_2d_cuda.h>
#include <oskar_dftw_o2c_3d_cuda.h>
#include <oskar_dftw_o2c_2d_omp.h>
#include <oskar_dftw_o2c_3d_omp.h>
#include <oskar_cuda_check_error.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_array_pattern(oskar_Mem* beam, double wavenumber,
        const oskar_Station* station, int num_points,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Mem* weights, int* status)
{
    int type, location, num_elements, array_is_3d;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    type = oskar_station_precision(station);
    location = oskar_station_mem_location(station);
    num_elements = oskar_station_num_elements(station);
    array_is_3d = oskar_station_array_is_3d(station);

    /* Check data are co-located. */
    if (oskar_mem_location(beam) != location ||
            oskar_mem_location(x) != location ||
            oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location ||
            oskar_mem_location(weights) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for correct data types. */
    if (!oskar_mem_is_complex(beam) ||
            !oskar_mem_is_complex(weights) ||
            oskar_mem_is_matrix(beam) ||
            oskar_mem_is_matrix(weights))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_type(x) != type || oskar_mem_type(y) != type ||
            oskar_mem_type(z) != type || oskar_mem_precision(beam) != type ||
            oskar_mem_precision(weights) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Resize output array if required. */
    if ((int)oskar_mem_length(beam) < num_points)
        oskar_mem_realloc(beam, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Switch on type and location. */
    if (type == OSKAR_DOUBLE)
    {
        const double *xs, *ys, *zs, *x_, *y_, *z_;
        const double2 *weights_;
        double2 *beam_;
        xs = oskar_mem_double_const(
                oskar_station_element_true_x_enu_metres_const(station), status);
        ys = oskar_mem_double_const(
                oskar_station_element_true_y_enu_metres_const(station), status);
        zs = oskar_mem_double_const(
                oskar_station_element_true_z_enu_metres_const(station), status);
        x_ = oskar_mem_double_const(x, status);
        y_ = oskar_mem_double_const(y, status);
        z_ = oskar_mem_double_const(z, status);
        weights_ = oskar_mem_double2_const(weights, status);
        beam_ = oskar_mem_double2(beam, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (array_is_3d)
            {
                oskar_dftw_o2c_3d_cuda_d(num_elements, wavenumber, xs, ys, zs,
                        weights_, num_points, x_, y_, z_, beam_);
            }
            else
            {
                oskar_dftw_o2c_2d_cuda_d(num_elements, wavenumber, xs, ys,
                        weights_, num_points, x_, y_, beam_);
            }
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            if (array_is_3d)
            {
                oskar_dftw_o2c_3d_omp_d(num_elements, wavenumber, xs, ys, zs,
                        weights_, num_points, x_, y_, z_, beam_);
            }
            else
            {
                oskar_dftw_o2c_2d_omp_d(num_elements, wavenumber, xs, ys,
                        weights_, num_points, x_, y_, beam_);
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *xs, *ys, *zs, *x_, *y_, *z_;
        const float2 *weights_;
        float2 *beam_;
        xs = oskar_mem_float_const(
                oskar_station_element_true_x_enu_metres_const(station), status);
        ys = oskar_mem_float_const(
                oskar_station_element_true_y_enu_metres_const(station), status);
        zs = oskar_mem_float_const(
                oskar_station_element_true_z_enu_metres_const(station), status);
        x_ = oskar_mem_float_const(x, status);
        y_ = oskar_mem_float_const(y, status);
        z_ = oskar_mem_float_const(z, status);
        weights_ = oskar_mem_float2_const(weights, status);
        beam_ = oskar_mem_float2(beam, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (array_is_3d)
            {
                oskar_dftw_o2c_3d_cuda_f(num_elements, wavenumber, xs, ys, zs,
                        weights_, num_points, x_, y_, z_, beam_);
            }
            else
            {
                oskar_dftw_o2c_2d_cuda_f(num_elements, wavenumber, xs, ys,
                        weights_, num_points, x_, y_, beam_);
            }
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            if (array_is_3d)
            {
                oskar_dftw_o2c_3d_omp_f(num_elements, wavenumber, xs, ys, zs,
                        weights_, num_points, x_, y_, z_, beam_);
            }
            else
            {
                oskar_dftw_o2c_2d_omp_f(num_elements, wavenumber, xs, ys,
                        weights_, num_points, x_, y_, beam_);
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
