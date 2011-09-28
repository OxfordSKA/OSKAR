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

#ifndef OSKAR_MS_APPEND_VIS_H_
#define OSKAR_MS_APPEND_VIS_H_

/**
 * @file oskar_ms_append_vis.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Adds the supplied visibilities and (u,v,w) coordinates to a Measurement Set.
 *
 * @details
 * This method adds the given block of visibility data to the main table of
 * the Measurement Set. The dimensionality of the complex \p vis data block
 * is \p n_pol x \p n_chan x \p n_row, with \p n_pol the fastest varying
 * dimension, then \p n_chan, and finally \p n_row.
 *
 * Each row of the main table holds data from a single baseline for a
 * single time stamp, so the number of rows is given by the number of
 * baselines multiplied by the number of times. The complex visibilities
 * are therefore understood to be given per polarisation, per channel and
 * per baseline (and repeated as many times as required).
 *
 * The times are given in units of (MJD) * 86400, i.e. seconds since
 * Julian date 2400000.5.
 *
 * Thus (for C-ordered memory), the layout of \p vis corresponding to two
 * time snapshots, for a three-element interferometer with four
 * polarisations and two channels would be:
 *
 * time0,ant0-1
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * time0,ant0-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * time0,ant1-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * time1,ant0-1
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * time1,ant0-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 * time1,ant1-2
 *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
 *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
 *
 * @param[in] ms_name The name of the Measurement Set directory.
 * @param[in] n_pol    Number of polarisations.
 * @param[in] n_chan   Number of channels.
 * @param[in] n_row    Number of rows to add to the main table (see note).
 * @param[in] u        Baseline u-coordinates, in metres (size n_row).
 * @param[in] v        Baseline v-coordinate, in metres (size n_row).
 * @param[in] w        Baseline w-coordinate, in metres (size n_row).
 * @param[in] vis      Matrix of complex visibilities per row (see note).
 * @param[in] ant1     Indices of antenna 1 for each baseline (size n_row).
 * @param[in] ant2     Indices of antenna 2 for each baseline (size n_row).
 * @param[in] exposure The exposure length per visibility, in seconds.
 * @param[in] interval The interval length per visibility, in seconds.
 * @param[in] times    Timestamp of each visibility block (size n_row).
 */
void oskar_ms_append_vis(const char* ms_name, int n_pol, int n_chan,
        int n_row, const double* u, const double* v, const double* w,
        const double* vis, const int* ant1, const int* ant2, double exposure,
        double interval, const double* times);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_MS_APPEND_VIS_H_
