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

#include "station/oskar_evaluate_e_jones_2d_horizontal.h"

#include "cuda/kernels/oskar_cudak_antenna.h"
#include "cuda/kernels/oskar_cudak_apodisation.h"
#include "cuda/kernels/oskar_cudak_bp2hiw.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"

#include "station/oskar_StationModel.h"
#include "sky/oskar_SkyModel.h"

#include "utility/oskar_cuda_eclipse.h"

#include "math/cudak/oskar_cudak_dftw_2d_seq_in.h"
#include "math/cudak/oskar_cudak_dftw_o2c_2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_evaluate_e_jones_2d_horizontal_f()
{
    cudaError_t cuda_error = cudaSuccess;

    return cuda_error;
}



// Double precision.
int oskar_evaluate_e_jones_2d_horizontal_d(
        const oskar_StationModel* hd_station,
        const double h_beam_l,
        const double h_beam_m,
        const oskar_SkyModelLocal_d* hd_sky,
        double2* d_weights_work,
        double2* d_e_jones
){
    // === Initialise.
    const int num_beams        = 1; // == 1 as this is a beam pattern!
    const int num_antennas     = hd_station->num_antennas;
    const double * d_antenna_x = hd_station->antenna_x;
    const double * d_antenna_y = hd_station->antenna_y;

    // === Invoke kernel to compute unnormalised, geometric antenna weights.
    int num_antennas_per_block = 256;
    dim3 block_dim(num_antennas_per_block, num_beams);  // (antennas, beams).
    dim3 grid_dim((num_antennas + block_dim.x - 1) / block_dim.x, 1);
    size_t shared_mem_size = (block_dim.x + block_dim.y) * sizeof(double2);

    // NOTE: maybe have version of weights generator passes by copy?
    double *d_beam_l, *d_beam_m; // beam position direction cosines.
    cudaMalloc(&d_beam_l, sizeof(double));
    cudaMalloc(&d_beam_m, sizeof(double));
    cudaMemcpy(d_beam_l, &h_beam_l, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beam_m, &h_beam_m, sizeof(double), cudaMemcpyHostToDevice);

    // === Generate dft weights.
    oskar_cudak_dftw_2d_seq_in_d<<<grid_dim, block_dim, shared_mem_size >>>
            (num_antennas, d_antenna_x, d_antenna_y, num_beams, d_beam_l,
                    d_beam_m, d_weights_work);

    // === Evaluate beam pattern for each source.
    const int naMax = 432;    // Should be multiple of 16.
    const int bThd = 256;     // Beam pattern generator (source positions).
    int bBlk = 0;             // Number of thread blocks for beam pattern computed later.
    bBlk = (hd_sky->num_sources + bThd - 1) / bThd;
    shared_mem_size = 2 * naMax * sizeof(double2);
    oskar_cudak_dftw_o2c_2d_d <<< bBlk, bThd, shared_mem_size >>>
            (num_antennas, d_antenna_x, d_antenna_y,
            d_weights_work, hd_sky->num_sources, hd_sky->hor_l, hd_sky->hor_m,
            naMax, d_e_jones);

    // === Free device memory.
    cudaFree(d_beam_l);
    cudaFree(d_beam_m);

    // === Check error code.
    cudaDeviceSynchronize();

    // === Return any CUDA error.
    return cudaPeekAtLastError();
}

#ifdef __cplusplus
}
#endif
