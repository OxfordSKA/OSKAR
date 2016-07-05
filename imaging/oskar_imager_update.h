/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_IMAGER_UPDATE_H_
#define OSKAR_IMAGER_UPDATE_H_

/**
 * @file oskar_imager_update.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_vis_block.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Intermediate-level function to run the imager and apply visibility selection.
 *
 * @details
 * This function updates the internal imager state using the
 * supplied visibilities.
 *
 * Visibility selection/filtering and phase rotation are performed
 * if necessary.
 *
 * @param[in,out] h             Handle to imager.
 * @param[in]     b             Handle to visibility block.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_imager_update_block(oskar_Imager* h, const oskar_VisBlock* b,
        int* status);

/**
 * @brief
 * Intermediate-level function to run the imager and apply visibility selection.
 *
 * @details
 * This function updates the internal imager state using the
 * supplied visibilities.
 *
 * Visibility selection/filtering and phase rotation are performed
 * if necessary.
 *
 * The baseline data dimension order must be
 * (slowest) time, baseline (fastest).
 *
 * The visibility amplitude data dimension order must be
 * (slowest) time, channel, baseline, polarisation (fastest).
 *
 * @param[in,out] h             Handle to imager.
 * @param[in]     start_time    Start time index of the visibility block.
 * @param[in]     end_time      End time index of the visibility block.
 * @param[in]     start_chan    Start channel index of the visibility block.
 * @param[in]     end_chan      End channel index of the visibility block.
 * @param[in]     num_pols      Number of polarisations in the visibility block.
 * @param[in]     num_baselines Number of baselines in the visibility block.
 * @param[in]     uu            Time-baseline ordered U-coordinates, in metres.
 * @param[in]     vv            Time-baseline ordered V-coordinates, in metres.
 * @param[in]     ww            Time-baseline ordered W-coordinates, in metres.
 * @param[in]     amps          Complex visibility amplitudes.
 * @param[in]     weight        Visibility weights.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_imager_update(oskar_Imager* h, int start_time, int end_time,
        int start_chan, int end_chan, int num_pols, int num_baselines,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, int* status);

/**
 * @brief
 * Low-level function to run the imager only for the supplied visibilities.
 *
 * @details
 * This low-level function runs one pass of the imager to update the
 * supplied plane for the supplied visibilities only.
 *
 * Visibility selection/filtering and phase rotation are
 * not available at this level.
 *
 * When using a DFT, \p plane refers to the image plane;
 * when using a FFT, \p plane refers to the visibility grid.
 *
 * The supplied baseline coordinates must be in wavelengths.
 *
 * @param[in,out] h             Handle to imager.
 * @param[in]     num_vis       Number of visibilities.
 * @param[in]     uu            Baseline uu coordinates, in wavelengths.
 * @param[in]     vv            Baseline vv coordinates, in wavelengths.
 * @param[in]     ww            Baseline ww coordinates, in wavelengths.
 * @param[in]     amps          Input complex visibilities.
 * @param[in]     weight        Visibility weights.
 * @param[in,out] plane         Updated image or visibility plane.
 * @param[in,out] plane_norm    Updated required normalisation of plane.
 * @param[in]     weights_grid  Updated grid of weights.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_imager_update_plane(oskar_Imager* h, int num_vis,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* amps, const oskar_Mem* weight, oskar_Mem* plane,
        double* plane_norm, const oskar_Mem* weights_grid, int* status);

/**
 * @brief
 * Updates the grid of weights.
 *
 * @details
 * This low-level function updates the grid of weights,
 * which is used during uniform weighting.
 * It also updates the statistics on W-coordinates,
 * which are used for W-projection.
 *
 * @param[in,out] h             Handle to imager.
 * @param[in]     num_points    Number of points.
 * @param[in]     uu            Baseline uu coordinates, in wavelengths.
 * @param[in]     vv            Baseline vv coordinates, in wavelengths.
 * @param[in]     ww            Baseline ww coordinates, in wavelengths.
 * @param[in]     weight        Visibility weights.
 * @param[in,out] weights_grid  Updated grid of weights.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_imager_update_weights_grid(oskar_Imager* h, int num_points,
        const oskar_Mem* uu, const oskar_Mem* vv, const oskar_Mem* ww,
        const oskar_Mem* weight, oskar_Mem* weights_grid, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGER_UPDATE_H_ */
