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

#include "oskar_global.h"

#include "station/oskar_evaluate_station_beam_scalar.h"
#include "utility/oskar_Mem.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_WorkE.h"
#include "utility/oskar_mem_element_size.h"
#include "math/cudak/oskar_cudak_dftw_2d.h"
#include "math/cudak/oskar_cudak_dftw_o2c_2d.h"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evalate_station_beam_scalar(oskar_Mem* E,
        const oskar_StationModel* station, oskar_WorkE* work)
{
    if (E == NULL || station == NULL || work == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    int num_antennas = station->n_elements;
    size_t element_size = oskar_mem_element_size(E->type());
    int num_sources = work->hor_l.n_elements();

    // Double precision.
    if (E->is_double() &&
            station->coord_type() == OSKAR_DOUBLE &&
            work->weights.is_double() &&
            work->hor_l.is_double() &&
            work->hor_m.is_double())
    {
        // DFT weights.
        int num_threads = 256;
        int num_blocks = (num_antennas + num_threads - 1) / num_threads;
        oskar_cudak_dftw_2d_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
                (num_antennas, station->x, station->y, work->beam_hor_l,
                        work->beam_hor_m, work->weights);

        // Evaluate beam pattern for each source.
        int antennas_per_chunk = 432;  // Should be multiple of 16.
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        size_t shared_mem_size = 2 * antennas_per_chunk * element_size;
        oskar_cudak_dftw_o2c_2d_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem_size)
                (num_antennas, station->x, station->y,
                work->weights, num_sources, work->hor_l, work->hor_m,
                antennas_per_chunk, (double2*)E->data);
    }


    // Single precision.
    else if (E->is_single() &&
            station->coord_type() == OSKAR_SINGLE &&
            work->weights.is_single() &&
            work->hor_l.is_single() &&
            work->hor_m.is_single())
    {
        // DFT weights.
        int num_threads = 256;
        int num_blocks = (num_antennas + num_threads - 1) / num_threads;
        oskar_cudak_dftw_2d_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
                (num_antennas, station->x, station->y, work->beam_hor_l,
                        work->beam_hor_m, work->weights);

        // Evaluate beam pattern for each source.
        int antennas_per_chunk = 864;  // Should be multiple of 16.
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        size_t shared_mem_size = 2 * antennas_per_chunk * element_size;
        oskar_cudak_dftw_o2c_2d_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem_size)
            (num_antennas, station->x, station->y, work->weights,
                    num_sources, work->hor_l, work->hor_m,
                    antennas_per_chunk, (float2*)E->data);
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    // Return any CUDA error.
    return cudaPeekAtLastError();
}

#ifdef __cplusplus
}
#endif
