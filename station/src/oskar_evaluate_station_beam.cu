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

#include "station/oskar_evaluate_station_beam.h"

#include "station/oskar_StationModel.h"
#include "utility/oskar_Mem.h"
#include "station/oskar_WorkE.h"
#include "station/oskar_evaluate_station_beam_scalar.h"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// These headers should be safe to remove when removing deprecated functions.
// --------------------------------------------------------------------------
#include "math/cudak/oskar_cudak_dftw_2d_seq_in.h"
#include "math/cudak/oskar_cudak_dftw_o2c_2d.h"
#include "math/cudak/oskar_cudak_vec_set_c.h"
#include "cuda/kernels/oskar_cudak_bp2hiw.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"
// --------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_station_beam(oskar_Mem* E, const oskar_StationModel* station,
        oskar_WorkE* work)
{
    if (E == NULL || station == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // NOTE extra fields will have to be added to these check
    // for element pattern data.

    if (E->is_null() || station->x.is_null() || station->y.is_null() ||
            work->hor_l.is_null() || work->hor_m.is_null() ||
            work->weights.is_null())
    {
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;
    }

    // Check that the relevant memory is on the GPU.
    if (E->location() != OSKAR_LOCATION_GPU ||
            station->coord_location() != OSKAR_LOCATION_GPU ||
            work->hor_l.location() != OSKAR_LOCATION_GPU ||
            work->hor_m.location() != OSKAR_LOCATION_GPU ||
            work->weights.location() != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (work->weights.n_elements() != station->n_elements ||
            work->hor_l.n_elements() != E->n_elements() ||
            work->hor_m.n_elements() != E->n_elements())
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (E->is_real() || work->weights.is_real())
        return OSKAR_ERR_BAD_DATA_TYPE;

    // No element pattern data. Assume isotropic antenna elements.
    if (station->element_pattern == NULL && E->is_scalar())
    {
        oskar_evalate_station_beam_scalar(E, station, work);
    }

    // Make use of element pattern data.
    else if (station->element_pattern != NULL && !E->is_scalar())
    {
        // NOTE Element pattern data ---> not yet implemented.
        return OSKAR_ERR_UNKNOWN;
    }
    else
    {
        return OSKAR_ERR_UNKNOWN;
    }

    return 0;
}


































// Double precision.
// DEPRECATED
int oskar_evaluate_station_beam_d(const oskar_StationModel_d* hd_station,
        const double h_beam_l, const double h_beam_m,
        const oskar_SkyModelLocal_d* hd_sky, double2* d_weights_work,
        double2* d_e_jones)
{
    // Initialise.
    const int num_beams       = 1; // == 1 as this is a beam pattern!
    const int num_antennas    = hd_station->num_antennas;
    const double* d_antenna_x = hd_station->antenna_x;
    const double* d_antenna_y = hd_station->antenna_y;

    // Invoke kernel to compute unnormalised, geometric antenna weights.
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

    // Generate dft weights.
    oskar_cudak_dftw_2d_seq_in_d<<<grid_dim, block_dim, shared_mem_size >>>
            (num_antennas, d_antenna_x, d_antenna_y, num_beams, d_beam_l,
                    d_beam_m, d_weights_work);

    // Evaluate beam pattern for each source.
    const int naMax = 432;    // Should be multiple of 16.
    const int bThd = 256;     // Beam pattern generator (source positions).
    int bBlk = 0;             // Number of thread blocks for beam pattern computed later.
    bBlk = (hd_sky->num_sources + bThd - 1) / bThd;
    shared_mem_size = 2 * naMax * sizeof(double2);
    oskar_cudak_dftw_o2c_2d_d <<< bBlk, bThd, shared_mem_size >>>
            (num_antennas, d_antenna_x, d_antenna_y,
            d_weights_work, hd_sky->num_sources, hd_sky->hor_l, hd_sky->hor_m,
            naMax, d_e_jones);

    // Free device memory.
    cudaFree(d_beam_l);
    cudaFree(d_beam_m);

    // Check error code.
    cudaDeviceSynchronize();

    // Return any CUDA error.
    return cudaPeekAtLastError();
}

// DEPRECATED
int oskar_evaluate_station_beam_f(const oskar_StationModel_f* hd_station,
        const float h_beam_l, const float h_beam_m,
        const oskar_SkyModelLocal_f* hd_sky, float2* d_weights_work,
        float2* d_e_jones)
{
    // Initialise.
      const int num_beams      = 1; // == 1 as this is a beam pattern!
      const int num_antennas   = hd_station->num_antennas;
      const float* d_antenna_x = hd_station->antenna_x;
      const float* d_antenna_y = hd_station->antenna_y;

      // Invoke kernel to compute unnormalised, geometric antenna weights.
      int num_antennas_per_block = 256;
      dim3 block_dim(num_antennas_per_block, num_beams);  // (antennas, beams).
      dim3 grid_dim((num_antennas + block_dim.x - 1) / block_dim.x, 1);
      size_t shared_mem_size = (block_dim.x + block_dim.y) * sizeof(float2);

      // NOTE: maybe have version of weights generator passes by copy?
      float *d_beam_l, *d_beam_m; // beam position direction cosines.
      cudaMalloc(&d_beam_l, sizeof(float));
      cudaMalloc(&d_beam_m, sizeof(float));
      cudaMemcpy(d_beam_l, &h_beam_l, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_beam_m, &h_beam_m, sizeof(float), cudaMemcpyHostToDevice);

      // Generate dft weights.
      oskar_cudak_dftw_2d_seq_in_f<<<grid_dim, block_dim, shared_mem_size >>>
              (num_antennas, d_antenna_x, d_antenna_y, num_beams, d_beam_l,
                      d_beam_m, d_weights_work);

      // Evaluate beam pattern for each source.
      const int naMax = 432;    // Should be multiple of 16.
      const int bThd = 256;     // Beam pattern generator (source positions).
      int bBlk = 0;             // Number of thread blocks for beam pattern computed later.
      bBlk = (hd_sky->num_sources + bThd - 1) / bThd;
      shared_mem_size = 2 * naMax * sizeof(float2);
      oskar_cudak_dftw_o2c_2d_f <<< bBlk, bThd, shared_mem_size >>>
              (num_antennas, d_antenna_x, d_antenna_y,
              d_weights_work, hd_sky->num_sources, hd_sky->hor_l, hd_sky->hor_m,
              naMax, d_e_jones);

      // Free device memory.
      cudaFree(d_beam_l);
      cudaFree(d_beam_m);

      // Check error code.
      cudaDeviceSynchronize();

      // Return any CUDA error.
      return cudaPeekAtLastError();
}


// DEPRECATED
void oskar_evaluate_station_beams_d(const unsigned num_stations,
        const oskar_StationModel_d* hd_stations,
        const oskar_SkyModelLocal_d* hd_sky, const double h_beam_l,
        const double h_beam_m, double2* d_weights_work,
        bool disable, bool identical_stations, double2* d_e_jones)
{
    // Station beam disabled.
    if (disable)
    {
        int num_threads = 128;
        int values = num_stations * hd_sky->num_sources;
        int num_blocks = (values + num_threads - 1) / num_threads;
        oskar_cudak_vec_set_c_d <<< num_blocks, num_threads >>>
                (values, make_double2(1.0, 0.0), d_e_jones);
    }
    // All stations identical.
    else if (identical_stations)
    {
        double2 * d_e_jones_station0 = d_e_jones;
        const oskar_StationModel_d * station0 = &hd_stations[0];
        oskar_evaluate_station_beam_d(station0, h_beam_l, h_beam_m,
                hd_sky, d_weights_work, d_e_jones_station0);
        for (unsigned i = 1; i < num_stations; ++i)
        {
            double2 * d_e_jones_station = d_e_jones + i * hd_sky->num_sources;
            cudaMemcpy(d_e_jones_station, d_e_jones_station0,
                    sizeof(double2) * hd_sky->num_sources,
                    cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        for (unsigned i = 0; i < num_stations; ++i)
        {
            double2 * d_e_jones_station = d_e_jones + i * hd_sky->num_sources;
            const oskar_StationModel_d * station = &hd_stations[i];
            oskar_evaluate_station_beam_d(station, h_beam_l, h_beam_m,
                    hd_sky, d_weights_work, d_e_jones_station);
        }
    }
}

// DEPRECATED
void oskar_evaluate_station_beams_f(const unsigned num_stations,
        const oskar_StationModel_f* hd_stations,
        const oskar_SkyModelLocal_f* hd_sky, const float h_beam_l,
        const float h_beam_m, float2* d_weights_work,
        bool disable, bool identical_stations, float2* d_e_jones)
{
    if (disable)
    {
        int num_threads = 128;
        int values = num_stations * hd_sky->num_sources;
        int num_blocks = (values + num_threads - 1) / num_threads;
        oskar_cudak_vec_set_c_f <<< num_blocks, num_threads >>>
                (values, make_float2(1.0, 0.0), d_e_jones);
    }
    else if (identical_stations)
    {
        float2 * d_e_jones_station0 = d_e_jones;
        const oskar_StationModel_f * station0 = &hd_stations[0];
        oskar_evaluate_station_beam_f(station0, h_beam_l, h_beam_m,
                hd_sky, d_weights_work, d_e_jones_station0);
        for (unsigned i = 1; i < num_stations; ++i)
        {
            float2 * d_e_jones_station = d_e_jones + i * hd_sky->num_sources;
            cudaMemcpy(d_e_jones_station, d_e_jones_station0,
                    sizeof(float2) * hd_sky->num_sources,
                    cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        for (unsigned i = 0; i < num_stations; ++i)
        {
            float2 * d_e_jones_station = d_e_jones + i * hd_sky->num_sources;
            const oskar_StationModel_f * station = &hd_stations[i];
            oskar_evaluate_station_beam_f(station, h_beam_l, h_beam_m,
                    hd_sky, d_weights_work, d_e_jones_station);
        }
    }
}



#ifdef __cplusplus
}
#endif
