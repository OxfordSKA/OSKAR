/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#ifndef OSKAR_MS_CREATE_H_
#define OSKAR_MS_CREATE_H_

/**
 * @file oskar_ms_create.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new Measurement Set.
 *
 * @details
 * Creates a new, empty Measurement Set with the given name.
 *
 * @param[in] file_name       The file name to use.
 * @param[in] app_name        The name of the application creating the MS.
 * @param[in] num_stations    The number of antennas/stations.
 * @param[in] num_channels    The number of channels in the band.
 * @param[in] num_pols        The number of polarisations (1, 2 or 4).
 * @param[in] freq_start_hz   The frequency at the centre of channel 0, in Hz.
 * @param[in] freq_inc_hz     The channel separation, in Hz.
 * @param[in] write_autocorr  If set, write auto-correlation data.
 * @param[in] write_crosscorr If set, write cross-correlation data.
 */
OSKAR_MS_EXPORT
oskar_MeasurementSet* oskar_ms_create(const char* file_name,
        const char* app_name, unsigned int num_stations,
        unsigned int num_channels, unsigned int num_pols, double freq_start_hz,
        double freq_inc_hz, int write_autocorr, int write_crosscorr);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MS_CREATE_H_ */
