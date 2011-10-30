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

#include "interferometry/oskar_evaluate_jones_K.h"
#include "interferometry/oskar_cuda_xyz_to_uvw.h"
#include "math/cudak/oskar_cudak_dftw_3d_seq_out.h"

extern "C"
int oskar_evaluate_jones_K(oskar_Jones* K, const oskar_SkyModel* sky,
        oskar_TelescopeModel* telescope, double gast)
{
    // Assert that the parameters are not NULL.
    if (K == NULL || sky == NULL || telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that the memory is not NULL.
    if (K->ptr.data == NULL || sky->rel_l.data == NULL ||
    		sky->rel_m.data == NULL || sky->rel_n.data == NULL ||
    		telescope->station_u.data == NULL ||
    		telescope->station_v.data == NULL ||
    		telescope->station_w.data == NULL ||
    		telescope->station_x.data == NULL ||
    		telescope->station_y.data == NULL ||
    		telescope->station_z.data == NULL)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    // Check that the data dimensions are OK.
    if (K->n_sources() != sky->num_sources ||
            K->n_stations() != telescope->num_stations)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check that the data is in the right location.
    if (K->location() != OSKAR_LOCATION_GPU ||
            sky->rel_l.location() != OSKAR_LOCATION_GPU ||
            sky->rel_m.location() != OSKAR_LOCATION_GPU ||
            sky->rel_n.location() != OSKAR_LOCATION_GPU ||
            telescope->station_u.location() != OSKAR_LOCATION_GPU ||
            telescope->station_v.location() != OSKAR_LOCATION_GPU ||
            telescope->station_w.location() != OSKAR_LOCATION_GPU ||
            telescope->station_x.location() != OSKAR_LOCATION_GPU ||
            telescope->station_y.location() != OSKAR_LOCATION_GPU ||
            telescope->station_z.location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check that the data is of the right type.
    if (K->type() == OSKAR_SINGLE_COMPLEX)
    {
        if (sky->rel_l.type() != OSKAR_SINGLE ||
        		sky->rel_m.type() != OSKAR_SINGLE ||
        		sky->rel_n.type() != OSKAR_SINGLE ||
        		telescope->station_u.type() != OSKAR_SINGLE ||
        		telescope->station_v.type() != OSKAR_SINGLE ||
        		telescope->station_w.type() != OSKAR_SINGLE ||
        		telescope->station_x.type() != OSKAR_SINGLE ||
        		telescope->station_y.type() != OSKAR_SINGLE ||
        		telescope->station_z.type() != OSKAR_SINGLE)
            return OSKAR_ERR_TYPE_MISMATCH;
    }
    else if (K->type() == OSKAR_DOUBLE_COMPLEX)
    {
        if (sky->rel_l.type() != OSKAR_DOUBLE ||
        		sky->rel_m.type() != OSKAR_DOUBLE ||
        		sky->rel_n.type() != OSKAR_DOUBLE ||
        		telescope->station_u.type() != OSKAR_DOUBLE ||
        		telescope->station_v.type() != OSKAR_DOUBLE ||
        		telescope->station_w.type() != OSKAR_DOUBLE ||
        		telescope->station_x.type() != OSKAR_DOUBLE ||
        		telescope->station_y.type() != OSKAR_DOUBLE ||
        		telescope->station_z.type() != OSKAR_DOUBLE)
            return OSKAR_ERR_TYPE_MISMATCH;
    }
    else
    {
        return OSKAR_ERR_BAD_JONES_TYPE;
    }

    // Get data sizes.
    int n_sources  = K->n_sources();
    int n_stations = K->n_stations();

    // Evaluate Jones matrix.
    if (K->type() == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
    	// Evaluate Greenwich Hour Angle of phase centre.
    	const float ha0 = (float)(gast - telescope->ra0);
    	const float dec0 = (float)telescope->dec0;

    	// Evaluate station u,v,w coordinates.
    	oskar_cuda_xyz_to_uvw_f(n_stations,
    			(const float*)telescope->station_x.data,
    			(const float*)telescope->station_y.data,
    			(const float*)telescope->station_z.data, ha0, dec0,
    			(float*)telescope->station_u.data,
    			(float*)telescope->station_v.data,
    			(float*)telescope->station_w.data);

        // Define block and grid sizes.
    	const dim3 n_thd(64, 4); // Sources, antennas.
    	const dim3 n_blk((n_sources + n_thd.x - 1) / n_thd.x,
        		(n_stations + n_thd.y - 1) / n_thd.y);
    	const size_t s_mem = 3 * (n_thd.x + n_thd.y) * sizeof(float);

    	// Compute DFT phase weights for K.
        oskar_cudak_dftw_3d_seq_out_f OSKAR_CUDAK_CONF(n_blk, n_thd, s_mem)
        (n_stations, (const float*)telescope->station_u.data,
        		(const float*)telescope->station_v.data,
        		(const float*)telescope->station_w.data, n_sources,
        		(const float*)sky->rel_l.data,
        		(const float*)sky->rel_m.data,
        		(const float*)sky->rel_n.data,
        		(float2*)K->ptr.data);
    }
    else if (K->type() == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
    	// Evaluate Greenwich Hour Angle of phase centre.
    	const double ha0 = gast - telescope->ra0;
    	const double dec0 = telescope->dec0;

    	// Evaluate station u,v,w coordinates.
    	oskar_cuda_xyz_to_uvw_d(n_stations,
    			(const double*)telescope->station_x.data,
    			(const double*)telescope->station_y.data,
    			(const double*)telescope->station_z.data, ha0, dec0,
    			(double*)telescope->station_u.data,
    			(double*)telescope->station_v.data,
    			(double*)telescope->station_w.data);

        // Define block and grid sizes.
        const dim3 n_thd(64, 4); // Sources, antennas.
        const dim3 n_blk((n_sources + n_thd.x - 1) / n_thd.x,
        		(n_stations + n_thd.y - 1) / n_thd.y);
        const size_t s_mem = 3 * (n_thd.x + n_thd.y) * sizeof(double);

    	// Compute DFT phase weights for K.
        oskar_cudak_dftw_3d_seq_out_d OSKAR_CUDAK_CONF(n_blk, n_thd, s_mem)
        (n_stations, (const double*)telescope->station_u.data,
        		(const double*)telescope->station_v.data,
        		(const double*)telescope->station_w.data, n_sources,
        		(const double*)sky->rel_l.data,
        		(const double*)sky->rel_m.data,
        		(const double*)sky->rel_n.data,
        		(double2*)K->ptr.data);
    }

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}
