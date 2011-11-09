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

int oskar_evalate_station_beam_scalar(oskar_Mem* beam,
        const oskar_StationModel* station, const double l_beam,
        const double m_beam, const oskar_Mem* l, const oskar_Mem* m,
        oskar_Mem* weights)
{
    if (beam == NULL || station == NULL || m == NULL ||
            l == NULL || weights == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (station->coord_units != OSKAR_WAVENUMBERS)
        return OSKAR_ERR_BAD_UNITS;

    // Resize the weights work array if needed.
    if (weights->num_elements() != station->num_elements)
    {
        int error = weights->resize(station->num_elements);
        if (error) return error;
    }

    int num_antennas = station->num_elements;
    size_t element_size = oskar_mem_element_size(beam->type());
    int num_sources = l->num_elements();

    // Double precision.
    if (beam->type() == OSKAR_DOUBLE_COMPLEX &&
            station->coord_type() == OSKAR_DOUBLE &&
            weights->type() == OSKAR_DOUBLE_COMPLEX &&
            l->type() == OSKAR_DOUBLE &&
            m->type() == OSKAR_DOUBLE)
    {
        // DFT weights.
        int num_threads = 256;
        int num_blocks = (num_antennas + num_threads - 1) / num_threads;
        oskar_cudak_dftw_2d_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
                (num_antennas, station->x, station->y, l_beam,
                        m_beam, *weights);

        // Evaluate beam pattern for each source.
        int antennas_per_chunk = 432;  // Should be multiple of 16.
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        size_t shared_mem_size = 2 * antennas_per_chunk * element_size;
        oskar_cudak_dftw_o2c_2d_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem_size)
                (num_antennas, station->x, station->y,
                *weights, num_sources, *l, *m, antennas_per_chunk, *beam);
    }


    // Single precision.
    else if (beam->type() == OSKAR_SINGLE_COMPLEX &&
            station->coord_type() == OSKAR_SINGLE &&
            weights->type() == OSKAR_SINGLE_COMPLEX &&
            l->type() == OSKAR_SINGLE &&
            m->type() == OSKAR_SINGLE)
    {
        // DFT weights.
        int num_threads = 256;
        int num_blocks = (num_antennas + num_threads - 1) / num_threads;
        oskar_cudak_dftw_2d_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
                (num_antennas, station->x, station->y, l_beam,
                        m_beam, *weights);

        // Evaluate beam pattern for each source.
        int antennas_per_chunk = 864;  // Should be multiple of 16.
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        size_t shared_mem_size = 2 * antennas_per_chunk * element_size;
        oskar_cudak_dftw_o2c_2d_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem_size)
            (num_antennas, station->x, station->y, *weights,
                    num_sources, *l, *m, antennas_per_chunk, *beam);
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
