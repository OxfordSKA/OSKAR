/*
 * Copyright (c) 2015, The University of Oxford
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

#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_auto_correlate(oskar_Mem* vis, int n_sources, const oskar_Jones* J,
        const oskar_Sky* sky, int* status)
{
    int jones_type, base_type, location, matrix_type, n_stations;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data dimensions. */
    n_stations = oskar_jones_num_stations(J);

    /* Check data locations. */
    location = oskar_sky_mem_location(sky);
    if (oskar_jones_mem_location(J) != location ||
            oskar_mem_location(vis) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for consistent data types. */
    jones_type = oskar_jones_type(J);
    base_type = oskar_sky_precision(sky);
    matrix_type = oskar_type_is_matrix(jones_type) &&
            oskar_mem_is_matrix(vis);
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
    if (oskar_jones_num_sources(J) < n_sources)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Select kernel. */
    if (base_type == OSKAR_DOUBLE)
    {
        const double *I_, *Q_, *U_, *V_;
        I_ = oskar_mem_double_const(oskar_sky_I_const(sky), status);
        Q_ = oskar_mem_double_const(oskar_sky_Q_const(sky), status);
        U_ = oskar_mem_double_const(oskar_sky_U_const(sky), status);
        V_ = oskar_mem_double_const(oskar_sky_V_const(sky), status);

        if (matrix_type)
        {
            double4c *vis_;
            const double4c *J_;
            vis_ = oskar_mem_double4c(vis, status);
            J_   = oskar_jones_double4c_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                oskar_auto_correlate_cuda_d(n_sources, n_stations,
                        J_, I_, Q_, U_, V_, vis_);
                oskar_device_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                oskar_auto_correlate_omp_d(n_sources, n_stations,
                        J_, I_, Q_, U_, V_, vis_);
            }
        }
        else /* Scalar version. */
        {
            double2 *vis_;
            const double2 *J_;
            vis_ = oskar_mem_double2(vis, status);
            J_   = oskar_jones_double2_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                oskar_auto_correlate_scalar_cuda_d(n_sources, n_stations,
                        J_, I_, vis_);
                oskar_device_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                oskar_auto_correlate_scalar_omp_d(n_sources, n_stations,
                        J_, I_, vis_);
            }
        }
    }
    else /* Single precision. */
    {
        const float *I_, *Q_, *U_, *V_;
        I_ = oskar_mem_float_const(oskar_sky_I_const(sky), status);
        Q_ = oskar_mem_float_const(oskar_sky_Q_const(sky), status);
        U_ = oskar_mem_float_const(oskar_sky_U_const(sky), status);
        V_ = oskar_mem_float_const(oskar_sky_V_const(sky), status);

        if (matrix_type)
        {
            float4c *vis_;
            const float4c *J_;
            vis_ = oskar_mem_float4c(vis, status);
            J_   = oskar_jones_float4c_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                oskar_auto_correlate_cuda_f(n_sources, n_stations,
                        J_, I_, Q_, U_, V_, vis_);
                oskar_device_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                oskar_auto_correlate_omp_f(n_sources, n_stations,
                        J_, I_, Q_, U_, V_, vis_);
            }
        }
        else /* Scalar version. */
        {
            float2 *vis_;
            const float2 *J_;
            vis_ = oskar_mem_float2(vis, status);
            J_   = oskar_jones_float2_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                oskar_auto_correlate_scalar_cuda_f(n_sources, n_stations,
                        J_, I_, vis_);
                oskar_device_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                oskar_auto_correlate_scalar_omp_f(n_sources, n_stations,
                        J_, I_, vis_);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
