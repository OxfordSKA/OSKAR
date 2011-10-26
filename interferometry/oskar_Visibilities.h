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

#ifndef OSKAR_VISIBILITIES_H_
#define OSKAR_VISIBILITIES_H_

/**
 * @file oskar_Visibilties.h
 */


#include "oskar_global.h"
#include "utility/oskar_Mem.h"


/**
 * @brief Structure to hold visibility data.
 *
 * @details
 *
 */
#ifdef __cplusplus
extern "C"
#endif
struct oskar_Visibilities
{
#ifdef __cplusplus
    public:
#endif
        int num_times;        ///< Number of time samples or visibility dumps.
        int num_baselines;    ///< Number of baselines.
        int num_channels;     ///< Number of frequency channels.
        oskar_Mem baseline_u; ///< Baseline coordinates, in wavenumbers.
        oskar_Mem baseline_v; ///< Baseline coordinates, in wavenumbers.
        oskar_Mem baseline_w; ///< Baseline coordinates, in wavenumbers.
        oskar_Mem amplitude;  ///< Complex visibility amplitude. Polarisation
                              ///< dimensions are specified by the type format.

    // Provide methods if C++.
#ifdef __cplusplus
    public:
        oskar_Visibilities();

        oskar_Visibilities(const int num_times, const int num_baselines,
                const int num_channels, const int amp_type, const int location);

        oskar_Visibilities(const char* filename);

        oskar_Visibilities(const oskar_Visibilities* other, const int location);

        ~oskar_Visibilities();

        int append(const oskar_Visibilities* other);

        int insert(const oskar_Visibilities* other, const unsigned time_index);

        int write(const char* filename);

        int read(const char* filename);

        int resize(int num_times, int num_baselines, int num_channels);

        int init(int num_times, int num_baselines, int num_channels,
                int amp_type, int location);

        int location() const { return amplitude.location(); }

        int num_samples() const { return num_times * num_baselines * num_channels; }

        int num_polarisations() const
        { return ((amplitude.type() & 0x0400) == 0x0400) ? 4 : 1; }

        int coord_type() const
        { return baseline_u.type(); }

        int amp_type() const
        { return amplitude.type(); }
#endif
};

typedef struct oskar_Visibilities oskar_Visibilities;


#endif // OSKAR_VISIBILITIES_H_
