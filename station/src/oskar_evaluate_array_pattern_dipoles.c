/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_evaluate_array_pattern_dipoles.h>
#include <oskar_evaluate_array_pattern_dipoles_cuda.h>
#include <oskar_cuda_check_error.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_array_pattern_dipoles(oskar_Mem* beam, double wavenumber,
        const oskar_Station* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, const oskar_Mem* weights,
        int* status)
{
    int type, location, num_elements;

    /* Check all inputs. */
    if (!beam || !station || !x || !y || !z || !weights || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    type = oskar_station_type(station);
    location = oskar_station_location(station);
    num_elements = oskar_station_num_elements(station);

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
    if (!oskar_mem_is_complex(beam) || !oskar_mem_is_complex(weights) ||
            !oskar_mem_is_matrix(beam) || oskar_mem_is_matrix(weights))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_type(x) != type || oskar_mem_type(y) != type ||
            oskar_mem_type(z) != type ||
            oskar_mem_precision(beam) != type ||
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

    /* Switch on type. */
    if (type == OSKAR_DOUBLE)
    {
        const double *xs, *ys, *zs, *cx, *sx, *cy, *sy, *x_, *y_, *z_;
        const double2 *weights_;
        double4c* beam_;
        xs = oskar_mem_double_const(
                oskar_station_element_x_signal_const(station), status);
        ys = oskar_mem_double_const(
                oskar_station_element_y_signal_const(station), status);
        zs = oskar_mem_double_const(
                oskar_station_element_z_signal_const(station), status);
        cx = oskar_mem_double_const(
                oskar_station_element_cos_orientation_x_const(station), status);
        sx = oskar_mem_double_const(
                oskar_station_element_sin_orientation_x_const(station), status);
        cy = oskar_mem_double_const(
                oskar_station_element_cos_orientation_y_const(station), status);
        sy = oskar_mem_double_const(
                oskar_station_element_sin_orientation_y_const(station), status);
        x_ = oskar_mem_double_const(x, status);
        y_ = oskar_mem_double_const(y, status);
        z_ = oskar_mem_double_const(z, status);
        weights_ = oskar_mem_double2_const(weights, status);
        beam_ = oskar_mem_double4c(beam, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_array_pattern_dipoles_cuda_d(num_elements,
                    wavenumber, xs, ys, zs, cx, sx, cy, sy, weights_,
                    num_points, x_, y_, z_, beam_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            /* TODO CPU version. */
            *status = OSKAR_ERR_BAD_LOCATION;
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *xs, *ys, *zs, *cx, *sx, *cy, *sy, *x_, *y_, *z_;
        const float2 *weights_;
        float4c* beam_;
        xs = oskar_mem_float_const(
                oskar_station_element_x_signal_const(station), status);
        ys = oskar_mem_float_const(
                oskar_station_element_y_signal_const(station), status);
        zs = oskar_mem_float_const(
                oskar_station_element_z_signal_const(station), status);
        cx = oskar_mem_float_const(
                oskar_station_element_cos_orientation_x_const(station), status);
        sx = oskar_mem_float_const(
                oskar_station_element_sin_orientation_x_const(station), status);
        cy = oskar_mem_float_const(
                oskar_station_element_cos_orientation_y_const(station), status);
        sy = oskar_mem_float_const(
                oskar_station_element_sin_orientation_y_const(station), status);
        x_ = oskar_mem_float_const(x, status);
        y_ = oskar_mem_float_const(y, status);
        z_ = oskar_mem_float_const(z, status);
        weights_ = oskar_mem_float2_const(weights, status);
        beam_ = oskar_mem_float4c(beam, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_array_pattern_dipoles_cuda_f(num_elements,
                    wavenumber, xs, ys, zs, cx, sx, cy, sy, weights_,
                    num_points, x_, y_, z_, beam_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            /* TODO CPU version. */
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
