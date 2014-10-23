/*
 * Copyright (c) 2014, The University of Oxford
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

#include <oskar_evaluate_average_cross_power_beam.h>
#include <oskar_evaluate_average_cross_power_beam_cuda.h>
#include <oskar_evaluate_average_cross_power_beam_omp.h>
#include <oskar_evaluate_average_scalar_cross_power_beam_cuda.h>
#include <oskar_evaluate_average_scalar_cross_power_beam_omp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper. */
void oskar_evaluate_average_cross_power_beam(int num_sources,
        int num_stations, const oskar_Jones* jones, oskar_Mem* beam,
        int *status)
{
    int type, location;

    /* Check all inputs. */
    if (!jones || !beam || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_jones_type(jones);
    location = oskar_jones_mem_location(jones);
    if (type != oskar_mem_type(beam))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(beam))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Switch on type and location combination. */
    if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_average_cross_power_beam_cuda_f(num_sources,
                    num_stations, oskar_jones_float4c_const(jones, status),
                    oskar_mem_float4c(beam, status));
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_evaluate_average_cross_power_beam_omp_f(num_sources,
                    num_stations, oskar_jones_float4c_const(jones, status),
                    oskar_mem_float4c(beam, status));
        }
    }
    else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_average_cross_power_beam_cuda_d(num_sources,
                    num_stations, oskar_jones_double4c_const(jones, status),
                    oskar_mem_double4c(beam, status));
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_evaluate_average_cross_power_beam_omp_d(num_sources,
                    num_stations, oskar_jones_double4c_const(jones, status),
                    oskar_mem_double4c(beam, status));
        }
    }

    /* Scalar versions. */
    else if (type == OSKAR_SINGLE_COMPLEX)
    {
        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_average_scalar_cross_power_beam_cuda_f(num_sources,
                    num_stations, oskar_jones_float2_const(jones, status),
                    oskar_mem_float2(beam, status));
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_evaluate_average_scalar_cross_power_beam_omp_f(num_sources,
                    num_stations, oskar_jones_float2_const(jones, status),
                    oskar_mem_float2(beam, status));
        }
    }
    else if (type == OSKAR_DOUBLE_COMPLEX)
    {
        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_average_scalar_cross_power_beam_cuda_d(num_sources,
                    num_stations, oskar_jones_double2_const(jones, status),
                    oskar_mem_double2(beam, status));
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_evaluate_average_scalar_cross_power_beam_omp_d(num_sources,
                    num_stations, oskar_jones_double2_const(jones, status),
                    oskar_mem_double2(beam, status));
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
