/*
 * Copyright (c) 2011, The University of Oxford
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

#include "sky/oskar_evaluate_jones_R.h"
#include "sky/cudak/oskar_cudak_evaluate_jones_R.h"
#include "utility/oskar_mem_element_size.h"

extern "C"
int oskar_evaluate_jones_R(oskar_Jones* R, const oskar_SkyModel* sky,
        const oskar_TelescopeModel* telescope, double gast)
{
    // Assert that the parameters are not NULL.
    if (R == NULL || sky == NULL || telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that the memory is not NULL.
    if (R->ptr.is_null() || sky->RA.is_null() || sky->Dec.is_null())
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    // Check that the data dimensions are OK.
    if (R->num_sources() != sky->num_sources ||
            R->num_stations() != telescope->num_stations)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check that the data is in the right location.
    if (R->location() != OSKAR_LOCATION_GPU ||
            sky->RA.location() != OSKAR_LOCATION_GPU ||
            sky->Dec.location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check that the data is of the right type.
    if (R->type() == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        if (sky->RA.type() != OSKAR_SINGLE || sky->Dec.type() != OSKAR_SINGLE)
            return OSKAR_ERR_TYPE_MISMATCH;
    }
    else if (R->type() == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        if (sky->RA.type() != OSKAR_DOUBLE || sky->Dec.type() != OSKAR_DOUBLE)
            return OSKAR_ERR_TYPE_MISMATCH;
    }
    else
    {
        return OSKAR_ERR_BAD_JONES_TYPE;
    }

    // Get data sizes.
    int n_sources  = R->num_sources();
    int n_stations = R->num_stations();

    // Define block and grid sizes.
    const int n_thd_f = 256;
    const int n_blk_f = (n_sources + n_thd_f - 1) / n_thd_f;
    const int n_thd_d = 256;
    const int n_blk_d = (n_sources + n_thd_d - 1) / n_thd_d;

    // Evaluate Jones matrix for each source for station 0.
    double cos_lat = cos(telescope->station[0].latitude);
    double sin_lat = sin(telescope->station[0].latitude);
    double lst = gast + telescope->station[0].longitude;
    if (R->type() == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        oskar_cudak_evaluate_jones_R_f OSKAR_CUDAK_CONF(n_blk_f, n_thd_f) (
                n_sources, sky->RA, sky->Dec, (float)cos_lat, (float)sin_lat,
                lst, R->ptr);
    }
    else if (R->type() == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        oskar_cudak_evaluate_jones_R_d OSKAR_CUDAK_CONF(n_blk_d, n_thd_d) (
                n_sources, sky->RA, sky->Dec, cos_lat, sin_lat, lst, R->ptr);
    }

    // Evaluate Jones matrix for each source for remaining stations.
    if (telescope->use_common_sky)
    {
        // Copy data for station 0 to stations 1 to n.
        cudaDeviceSynchronize();
        void* start = R->ptr.data;

        if (R->type() == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            for (int i = 1; i < n_stations; ++i)
                cudaMemcpy((float4c*)(R->ptr.data) + i * n_sources, start,
                        oskar_mem_element_size(R->type()) * n_sources,
                        cudaMemcpyDeviceToDevice);
        }
        else if (R->type() == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            for (int i = 1; i < n_stations; ++i)
                cudaMemcpy((double4c*)(R->ptr.data) + i * n_sources, start,
                        oskar_mem_element_size(R->type()) * n_sources,
                        cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        // Evaluate Jones matrix for each source for remaining stations.
        for (int i = 1; i < n_stations; ++i)
        {
            // Get station location.
            double cos_lat = cos(telescope->station[i].latitude);
            double sin_lat = sin(telescope->station[i].latitude);
            double lst = gast + telescope->station[i].longitude;

            if (R->type() == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                oskar_cudak_evaluate_jones_R_f
                OSKAR_CUDAK_CONF(n_blk_f, n_thd_f) (n_sources, sky->RA,
                        sky->Dec, (float)cos_lat, (float)sin_lat, lst,
                        (float4c*)(R->ptr.data) + i * n_sources);
            }
            else if (R->type() == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                oskar_cudak_evaluate_jones_R_d
                OSKAR_CUDAK_CONF(n_blk_d, n_thd_d) (n_sources, sky->RA,
                        sky->Dec, cos_lat, sin_lat, lst,
                        (double4c*)(R->ptr.data) + i * n_sources);
            }
        }
    }

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}
