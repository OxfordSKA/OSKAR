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

#include "interferometry/oskar_cuda_interferometer1_scalar.h"
#include "interferometry/oskar_cuda_correlator_scalar.h"

#include "math/cudak/oskar_math_cudak_dftw_3d_seq_out.h"
#include "math/cudak/oskar_math_cudak_mat_mul_cc.h"
#include "interferometry/cudak/oskar_cudak_correlator.h"
#include "interferometry/cudak/oskar_cudak_xyz2uvw.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEC_PER_DAY 86400.0

int oskar_cudad_interferometer1_scalar(

        const unsigned * num_antennas,
        const float * antenna_x, // FIXME: mmm better way to represent this
        const float * antenna_z, // === make sure it links to matlab.
        const float * antenna_y,

        const unsigned num_stations,
        const float * station_x,
        const float * station_y,
        const float * station_z,

        const unsigned num_sources,
        const float * source_l,
        const float * source_m,
        const float * source_n,
        const float * eb,

        const float ra0,
        const float dec0,

        const float start_date_utc,
        const unsigned nsdt,
        const float sdt,

        const float lambda_bandwidth,

        float * vis,
        float * work)
{
    cudaError_t cuda_error = cudaSuccess;

    // 1. Allocate GPU memory for antennas and transfer to the device.
    // TODO

    // 2. Allocate GPU memory for stations and transfer to device.
    // TODO

    // 3. Allocate GPU memory for source model and transfer to device.
    // TODO


    // 4. Loop over number of visibility snapshots.
    for (int j = 0; j < num_vis_snapshots; ++j)
    {
        // 5. Evaluate LST from UTC.
        // TODO

        // 6. Loop over evaluations of the visibility average with changing E-Jones
        // within the dump
        for (int i = 0; i < num_vis_averages; ++i)
        {
            // 6. Find sources above horizon.
            // TODO

            // 7. Evaluate E-Jones for each source position per station.
            // TODO: optimisation if all stations are the same?
            //for each station
            //      oskar_cudad_bp2hc() <=== move this to beamforming folder.

            // 8. Correlator which updates phase matrix.
            // TODO
            // oskar_cudad_correlator_sclar()

            // 9. Accumulate visibilities.
            // TODO
        }

        // 10. Dump a new set of visibilities including baseline coordinates.
        // TODO
    }


    return cuda_error;
}



#ifdef __cplusplus
}
#endif
