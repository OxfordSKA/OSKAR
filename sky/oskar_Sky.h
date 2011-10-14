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

#ifndef OSKAR_SKY_STRUCT_H_
#define OSKAR_SKY_STRUCT_H_

#ifdef __cplusplus
extern "C"
#endif
struct oskar_Sky
{
    int num_sources;
    oskar_Ptr RA;
    oskar_Ptr Dec;
    oskar_Ptr I;
    oskar_Ptr Q;
    oskar_Ptr U;
    oskar_Ptr V;
    oskar_Ptr reference_freq;
    oskar_Ptr spectral_index;

    // Work buffers.
    // NOTE: need better name to indicate they should be treated as work buffers.
    double update_timestamp; ///< Time for which work buffer is valid.
    oskar_Ptr rel_l;  ///< Phase centre relative direction-cosines.
    oskar_Ptr rel_m;  ///< Phase centre relative direction-cosines.
    oskar_Ptr rel_n;  ///< Phase centre relative direction-cosines.
    oskar_Ptr hor_l;  ///< Horizontal coordinate system direction-cosines.
    oskar_Ptr hor_m;  ///< Horizontal coordinate system direction-cosines.
    oskar_Ptr hor_n;  ///< Horizontal coordinate system direction-cosines.
};
typedef struct oskar_Sky oskar_Sky;

#endif // OSKAR_SKY_STRUCT_H_
