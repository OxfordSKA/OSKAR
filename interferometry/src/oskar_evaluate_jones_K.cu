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
#include "interferometry/oskar_xyz_to_uvw_cuda.h"
#include "math/cudak/oskar_cudak_dftw_3d_seq_out.h"

extern "C"
int oskar_evaluate_jones_K(oskar_Jones* K, const oskar_SkyModel* sky,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w)
{
    // Assert that the parameters are not NULL.
    if (K == NULL || sky == NULL || u == NULL || v == NULL || w == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that the memory is not NULL.
    if (K->ptr.is_null() || sky->rel_l.is_null() ||
            sky->rel_m.is_null() || sky->rel_n.is_null() ||
            u->is_null() || v->is_null() || w->is_null())
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    // Check that the data dimensions are OK.
    if (K->num_sources() != sky->num_sources ||
            K->num_stations() != u->num_elements() ||
            K->num_stations() != v->num_elements() ||
            K->num_stations() != w->num_elements())
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check that the data is in the right location.
    if (K->location() != OSKAR_LOCATION_GPU ||
            sky->rel_l.location() != OSKAR_LOCATION_GPU ||
            sky->rel_m.location() != OSKAR_LOCATION_GPU ||
            sky->rel_n.location() != OSKAR_LOCATION_GPU ||
            u->location() != OSKAR_LOCATION_GPU ||
            v->location() != OSKAR_LOCATION_GPU ||
            w->location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check that the data is of the right type.
    if (K->type() == OSKAR_SINGLE_COMPLEX)
    {
        if (sky->rel_l.type() != OSKAR_SINGLE ||
                sky->rel_m.type() != OSKAR_SINGLE ||
                sky->rel_n.type() != OSKAR_SINGLE ||
                u->type() != OSKAR_SINGLE ||
                v->type() != OSKAR_SINGLE ||
                w->type() != OSKAR_SINGLE)
            return OSKAR_ERR_TYPE_MISMATCH;
    }
    else if (K->type() == OSKAR_DOUBLE_COMPLEX)
    {
        if (sky->rel_l.type() != OSKAR_DOUBLE ||
                sky->rel_m.type() != OSKAR_DOUBLE ||
                sky->rel_n.type() != OSKAR_DOUBLE ||
                u->type() != OSKAR_DOUBLE ||
                v->type() != OSKAR_DOUBLE ||
                w->type() != OSKAR_DOUBLE)
            return OSKAR_ERR_TYPE_MISMATCH;
    }
    else
    {
        return OSKAR_ERR_BAD_JONES_TYPE;
    }

    // Get data sizes.
    const int n_sources  = K->num_sources();
    const int n_stations = K->num_stations();

    // Evaluate Jones matrix.
    if (K->type() == OSKAR_SINGLE_COMPLEX)
    {
        // Define block and grid sizes.
        const dim3 n_thd(64, 4); // Sources, antennas.
        const dim3 n_blk((n_sources + n_thd.x - 1) / n_thd.x,
                (n_stations + n_thd.y - 1) / n_thd.y);
        const size_t s_mem = 3 * (n_thd.x + n_thd.y) * sizeof(float);

        // Compute DFT phase weights for K.
        oskar_cudak_dftw_3d_seq_out_f OSKAR_CUDAK_CONF(n_blk, n_thd, s_mem)
        (n_stations, *u, *v, *w, n_sources, sky->rel_l, sky->rel_m, sky->rel_n,
                K->ptr);
    }
    else if (K->type() == OSKAR_DOUBLE_COMPLEX)
    {
        // Define block and grid sizes.
        const dim3 n_thd(64, 4); // Sources, antennas.
        const dim3 n_blk((n_sources + n_thd.x - 1) / n_thd.x,
                (n_stations + n_thd.y - 1) / n_thd.y);
        const size_t s_mem = 3 * (n_thd.x + n_thd.y) * sizeof(double);

        // Compute DFT phase weights for K.
        oskar_cudak_dftw_3d_seq_out_d OSKAR_CUDAK_CONF(n_blk, n_thd, s_mem)
        (n_stations, *u, *v, *w, n_sources, sky->rel_l, sky->rel_m, sky->rel_n,
                K->ptr);
    }
    else
    {
        return OSKAR_ERR_BAD_JONES_TYPE;
    }

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}
