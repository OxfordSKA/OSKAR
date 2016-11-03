/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#ifndef OSKAR_MS_WRITE_H_
#define OSKAR_MS_WRITE_H_

/**
 * @file oskar_ms_write.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Writes baseline coordinate data to the main table.
 *
 * @details
 * This function writes the supplied list of baseline coordinates to
 * the main table of the Measurement Set, extending it if necessary.
 *
 * Baseline antenna-pair ordering is implicit:
 * a0-a1, a0-a2, a0-a3... a1-a2, a1-a3... a2-a3 etc.
 * The supplied number of baselines must be compatible with the number of
 * stations in the Measurement Set. Auto-correlations are allowed.
 *
 * This function should be called for each time step to write out the
 * baseline coordinate data.
 *
 * The time stamp is given in units of (MJD) * 86400, i.e. seconds since
 * Julian date 2400000.5.
 *
 * @param[in] start_row     The start row index to write (zero-based).
 * @param[in] num_baselines Number of rows to write to the main table.
 * @param[in] uu            Baseline u-coordinates, in metres.
 * @param[in] vv            Baseline v-coordinates, in metres.
 * @param[in] ww            Baseline w-coordinates, in metres.
 * @param[in] exposure_sec  The exposure length per visibility, in seconds.
 * @param[in] interval_sec  The interval length per visibility, in seconds.
 * @param[in] time_stamp    Time stamp of coordinate data.
 */
OSKAR_MS_EXPORT
void oskar_ms_write_coords_d(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const double* uu, const double* vv, const double* ww,
        double exposure_sec, double interval_sec, double time_stamp);

/**
 * @details
 * Writes baseline coordinate data to the main table.
 *
 * @details
 * This function writes the supplied list of baseline coordinates to
 * the main table of the Measurement Set, extending it if necessary.
 *
 * Baseline antenna-pair ordering is implicit:
 * a0-a1, a0-a2, a0-a3... a1-a2, a1-a3... a2-a3 etc.
 * The supplied number of baselines must be compatible with the number of
 * stations in the Measurement Set. Auto-correlations are allowed.
 *
 * This function should be called for each time step to write out the
 * baseline coordinate data.
 *
 * The time stamp is given in units of (MJD) * 86400, i.e. seconds since
 * Julian date 2400000.5.
 *
 * @param[in] start_row     The start row index to write (zero-based).
 * @param[in] num_baselines Number of rows to write to the main table.
 * @param[in] uu            Baseline u-coordinates, in metres.
 * @param[in] vv            Baseline v-coordinates, in metres.
 * @param[in] ww            Baseline w-coordinates, in metres.
 * @param[in] exposure_sec  The exposure length per visibility, in seconds.
 * @param[in] interval_sec  The interval length per visibility, in seconds.
 * @param[in] time_stamp    Time stamp of coordinate data.
 */
OSKAR_MS_EXPORT
void oskar_ms_write_coords_f(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const float* uu, const float* vv, const float* ww,
        double exposure_sec, double interval_sec, double time_stamp);

/**
 * @details
 * Writes visibility data to the main table.
 *
 * @details
 * This function writes the given block of visibility data to the
 * data column of the Measurement Set, extending it if necessary.
 *
 * Baseline antenna-pair ordering is implicit:
 * a0-a1, a0-a2, a0-a3... a1-a2, a1-a3... a2-a3 etc.
 * The supplied number of baselines must be compatible with the number of
 * stations in the Measurement Set. Auto-correlations are allowed.
 *
 * This function should be called for each time step to write out the
 * visibility data.
 *
 * The dimensionality of the complex \p vis data block is:
 * (num_channels * num_baselines * num_pols),
 * with num_pols the fastest varying dimension, then num_baselines,
 * and num_channels the slowest.
 *
 * @param[in] start_row     The start row index to write (zero-based).
 * @param[in] start_channel The start channel index of the visibility block.
 * @param[in] num_channels  The number of channels in the visibility block.
 * @param[in] num_baselines The number of baselines in the visibility block.
 * @param[in] vis           Pointer to complex visibility block.
 */
OSKAR_MS_EXPORT
void oskar_ms_write_vis_d(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines,
        const double* vis);

/**
 * @details
 * Writes visibility data to the main table.
 *
 * @details
 * This function writes the given block of visibility data to the
 * data column of the Measurement Set, extending it if necessary.
 *
 * Baseline antenna-pair ordering is implicit:
 * a0-a1, a0-a2, a0-a3... a1-a2, a1-a3... a2-a3 etc.
 * The supplied number of baselines must be compatible with the number of
 * stations in the Measurement Set. Auto-correlations are allowed.
 *
 * This function should be called for each time step to write out the
 * visibility data.
 *
 * The dimensionality of the complex \p vis data block is:
 * (num_channels * num_baselines * num_pols),
 * with num_pols the fastest varying dimension, then num_baselines,
 * and num_channels the slowest.
 *
 * @param[in] start_row     The start row index to write (zero-based).
 * @param[in] start_channel The start channel index of the visibility block.
 * @param[in] num_channels  The number of channels in the visibility block.
 * @param[in] num_baselines The number of baselines in the visibility block.
 * @param[in] vis           Pointer to complex visibility block.
 */
OSKAR_MS_EXPORT
void oskar_ms_write_vis_f(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines,
        const float* vis);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MS_WRITE_H_ */
