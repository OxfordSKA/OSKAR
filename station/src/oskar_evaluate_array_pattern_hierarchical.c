/*
 * Copyright (c) 2013, The University of Oxford
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

#include "math/oskar_dftw_c2c_2d_cuda.h"
#include "math/oskar_dftw_c2c_3d_cuda.h"
#include "math/oskar_dftw_m2m_2d_cuda.h"
#include "math/oskar_dftw_m2m_3d_cuda.h"
#include "math/oskar_dftw_c2c_2d_omp.h"
#include "math/oskar_dftw_c2c_3d_omp.h"
#include "math/oskar_dftw_m2m_2d_omp.h"
#include "math/oskar_dftw_m2m_3d_omp.h"
#include "station/oskar_evaluate_array_pattern_hierarchical.h"
#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_type.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_array_pattern_hierarchical(oskar_Mem* beam,
        const oskar_StationModel* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, const oskar_Mem* signal,
        const oskar_Mem* weights, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!beam || !station || !x || !y || !z || !signal || !weights || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    location = oskar_station_model_location(station);
    type = oskar_station_model_type(station);

    /* Check data are co-located. */
    if (beam->location != location ||
            x->location != location ||
            y->location != location ||
            z->location != location ||
            signal->location != location ||
            weights->location != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that the antenna coordinates are in radians. */
    if (station->coord_units != OSKAR_RADIANS)
    {
        *status = OSKAR_ERR_BAD_UNITS;
        return;
    }

    /* Check for correct data types. */
    if (!oskar_mem_is_complex(beam->type) ||
            !oskar_mem_is_complex(signal->type) ||
            !oskar_mem_is_complex(weights->type) ||
            oskar_mem_is_matrix(weights->type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    if (x->type != type || y->type != type || z->type != type ||
            oskar_mem_base_type(beam->type) != type ||
            oskar_mem_base_type(signal->type) != type ||
            oskar_mem_base_type(weights->type) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Resize output array if required. */
    if (beam->num_elements < num_points)
        oskar_mem_realloc(beam, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check for data in GPU memory. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            if (oskar_mem_is_matrix(beam->type) &&
                    oskar_mem_is_matrix(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_m2m_3d_cuda_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double*)station->z_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double*)z->data,
                            (const double4c*)signal->data,
                            (double4c*)beam->data);
                }
                else
                {
                    oskar_dftw_m2m_2d_cuda_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double4c*)signal->data,
                            (double4c*)beam->data);
                }
                oskar_cuda_check_error(status);
            }
            else if (oskar_mem_is_scalar(beam->type) &&
                    oskar_mem_is_scalar(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_c2c_3d_cuda_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double*)station->z_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double*)z->data,
                            (const double2*)signal->data,
                            (double2*)beam->data);
                }
                else
                {
                    oskar_dftw_c2c_2d_cuda_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double2*)signal->data,
                            (double2*)beam->data);
                }
                oskar_cuda_check_error(status);
            }
            else
            {
                *status = OSKAR_ERR_TYPE_MISMATCH;
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            if (oskar_mem_is_matrix(beam->type) &&
                    oskar_mem_is_matrix(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_m2m_3d_cuda_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float*)station->z_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float*)z->data,
                            (const float4c*)signal->data,
                            (float4c*)beam->data);
                }
                else
                {
                    oskar_dftw_m2m_2d_cuda_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float4c*)signal->data,
                            (float4c*)beam->data);
                }
                oskar_cuda_check_error(status);
            }
            else if (oskar_mem_is_scalar(beam->type) &&
                    oskar_mem_is_scalar(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_c2c_3d_cuda_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float*)station->z_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float*)z->data,
                            (const float2*)signal->data,
                            (float2*)beam->data);
                }
                else
                {
                    oskar_dftw_c2c_2d_cuda_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float2*)signal->data,
                            (float2*)beam->data);
                }
                oskar_cuda_check_error(status);
            }
            else
            {
                *status = OSKAR_ERR_TYPE_MISMATCH;
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else /* OpenMP version. */
    {
        if (type == OSKAR_DOUBLE)
        {
            if (oskar_mem_is_matrix(beam->type) &&
                    oskar_mem_is_matrix(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_m2m_3d_omp_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double*)station->z_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double*)z->data,
                            (const double4c*)signal->data,
                            (double4c*)beam->data);
                }
                else
                {
                    oskar_dftw_m2m_2d_omp_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double4c*)signal->data,
                            (double4c*)beam->data);
                }
            }
            else if (oskar_mem_is_scalar(beam->type) &&
                    oskar_mem_is_scalar(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_c2c_3d_omp_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double*)station->z_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double*)z->data,
                            (const double2*)signal->data,
                            (double2*)beam->data);
                }
                else
                {
                    oskar_dftw_c2c_2d_omp_d(station->num_elements,
                            (const double*)station->x_signal.data,
                            (const double*)station->y_signal.data,
                            (const double2*)weights->data, num_points,
                            (const double*)x->data,
                            (const double*)y->data,
                            (const double2*)signal->data,
                            (double2*)beam->data);
                }
            }
            else
            {
                *status = OSKAR_ERR_TYPE_MISMATCH;
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            if (oskar_mem_is_matrix(beam->type) &&
                    oskar_mem_is_matrix(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_m2m_3d_omp_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float*)station->z_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float*)z->data,
                            (const float4c*)signal->data,
                            (float4c*)beam->data);
                }
                else
                {
                    oskar_dftw_m2m_2d_omp_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float4c*)signal->data,
                            (float4c*)beam->data);
                }
            }
            else if (oskar_mem_is_scalar(beam->type) &&
                    oskar_mem_is_scalar(signal->type))
            {
                if (station->array_is_3d)
                {
                    oskar_dftw_c2c_3d_omp_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float*)station->z_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float*)z->data,
                            (const float2*)signal->data,
                            (float2*)beam->data);
                }
                else
                {
                    oskar_dftw_c2c_2d_omp_f(station->num_elements,
                            (const float*)station->x_signal.data,
                            (const float*)station->y_signal.data,
                            (const float2*)weights->data, num_points,
                            (const float*)x->data,
                            (const float*)y->data,
                            (const float2*)signal->data,
                            (float2*)beam->data);
                }
            }
            else
            {
                *status = OSKAR_ERR_TYPE_MISMATCH;
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
}

#ifdef __cplusplus
}
#endif
