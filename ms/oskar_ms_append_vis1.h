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

#ifndef OSKAR_MS_APPEND_VIS1_H_
#define OSKAR_MS_APPEND_VIS1_H_

/**
 * @file oskar_ms_append_vis1.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Adds the supplied visibilities and (u,v,w) coordinates to a Measurement Set.
 *
 * @details
 * This functions appends to a Measurement Set using the supplied visibilities
 * and (u,v,w) coordinates for a single polarisation.
 *
 * @param[in] name The name of the Measurement Set directory.
 * @param[in] mjd The Modified Julian Date of the start time.
 * @param[in] exposure The visibility exposure length in seconds.
 * @param[in] interval The visibility interval length in seconds.
 * @param[in] nv The number of visibilities to add.
 * @param[in] u The visibility u coordinates in metres (length nv).
 * @param[in] v The visibility v coordinates in metres (length nv).
 * @param[in] w The visibility w coordinates in metres (length nv).
 * @param[in] vis The complex visibility values (length 2*nv).
 * @param[in] ant1 The index of antenna 1 for the baselines (length nv).
 * @param[in] ant2 The index of antenna 2 for the baselines (length nv).
 */
void oskar_ms_append_vis1(const char* name, double mjd, double exposure,
        double interval, int nv, const float* u, const float* v,
        const float* w, const float* vis, const int* ant1, const int* ant2);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_MS_APPEND_VIS1_H_
