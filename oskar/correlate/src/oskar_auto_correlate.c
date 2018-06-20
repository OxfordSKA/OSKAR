/*
 * Copyright (c) 2015-2018, The University of Oxford
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

#include "correlate/oskar_auto_correlate.h"
#include "correlate/oskar_auto_correlate_cuda.h"
#include "correlate/oskar_auto_correlate_omp.h"
#include "correlate/oskar_auto_correlate_scalar_cuda.h"
#include "correlate/oskar_auto_correlate_scalar_omp.h"
#include "utility/oskar_device_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_auto_correlate(oskar_Mem* vis, int n_sources,
        const oskar_Jones* jones, const oskar_Sky* sky, int* status)
{
    int jones_type, base_type, location, n_stations;
    const oskar_Mem *J, *I, *Q, *U, *V;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data dimensions. */
    n_stations = oskar_jones_num_stations(jones);

    /* Check data locations. */
    location = oskar_sky_mem_location(sky);
    if (oskar_jones_mem_location(jones) != location ||
            oskar_mem_location(vis) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for consistent data types. */
    jones_type = oskar_jones_type(jones);
    base_type = oskar_sky_precision(sky);
    if (oskar_mem_precision(vis) != base_type ||
            oskar_type_precision(jones_type) != base_type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_type(vis) != jones_type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* If neither single or double precision, return error. */
    if (base_type != OSKAR_SINGLE && base_type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check the input dimensions. */
    if (oskar_jones_num_sources(jones) < n_sources)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get handles to arrays. */
    J = oskar_jones_mem_const(jones);
    I = oskar_sky_I_const(sky);
    Q = oskar_sky_Q_const(sky);
    U = oskar_sky_U_const(sky);
    V = oskar_sky_V_const(sky);

    /* Select kernel. */
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(vis))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_auto_correlate_omp_f(n_sources, n_stations,
                    oskar_mem_float4c_const(J, status),
                    oskar_mem_float_const(I, status),
                    oskar_mem_float_const(Q, status),
                    oskar_mem_float_const(U, status),
                    oskar_mem_float_const(V, status),
                    oskar_mem_float4c(vis, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_auto_correlate_omp_d(n_sources, n_stations,
                    oskar_mem_double4c_const(J, status),
                    oskar_mem_double_const(I, status),
                    oskar_mem_double_const(Q, status),
                    oskar_mem_double_const(U, status),
                    oskar_mem_double_const(V, status),
                    oskar_mem_double4c(vis, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            oskar_auto_correlate_scalar_omp_f(n_sources, n_stations,
                    oskar_mem_float2_const(J, status),
                    oskar_mem_float_const(I, status),
                    oskar_mem_float2(vis, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_auto_correlate_scalar_omp_d(n_sources, n_stations,
                    oskar_mem_double2_const(J, status),
                    oskar_mem_double_const(I, status),
                    oskar_mem_double2(vis, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        switch (oskar_mem_type(vis))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_auto_correlate_cuda_f(n_sources, n_stations,
                    oskar_mem_float4c_const(J, status),
                    oskar_mem_float_const(I, status),
                    oskar_mem_float_const(Q, status),
                    oskar_mem_float_const(U, status),
                    oskar_mem_float_const(V, status),
                    oskar_mem_float4c(vis, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_auto_correlate_cuda_d(n_sources, n_stations,
                    oskar_mem_double4c_const(J, status),
                    oskar_mem_double_const(I, status),
                    oskar_mem_double_const(Q, status),
                    oskar_mem_double_const(U, status),
                    oskar_mem_double_const(V, status),
                    oskar_mem_double4c(vis, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            oskar_auto_correlate_scalar_cuda_f(n_sources, n_stations,
                    oskar_mem_float2_const(J, status),
                    oskar_mem_float_const(I, status),
                    oskar_mem_float2(vis, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_auto_correlate_scalar_cuda_d(n_sources, n_stations,
                    oskar_mem_double2_const(J, status),
                    oskar_mem_double_const(I, status),
                    oskar_mem_double2(vis, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
