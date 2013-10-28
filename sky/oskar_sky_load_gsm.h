/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_SKY_LOAD_GSM_H_
#define OSKAR_SKY_LOAD_GSM_H_

/**
 * @file oskar_sky_load_gsm.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads pixels from a GSM text file into an OSKAR sky model structure.
 *
 * @details
 * This function loads data from a GSM text file.
 * The data must be given in the HEALPix RING scheme.
 *
 * N.B. This function assumes that the values of pixels in the input GSM
 * data file are in units of Kelvin per steradian. These are converted to
 * Jansky per pixel using the following method:
 *
 * First obtain Kelvin per pixel by dividing the values of the input points
 * by the number of pixels per steradian.
 *
 * Then convert Kelvin per pixel to Jansky per pixel using the relation
 * between antenna temperature T and flux S:
 * S(Jy) = 2 * k_B * T(K) * 1e26.
 *
 * Note that this assumes that any wavelength dependence is already
 * in the input temperature data, so there is NO division by the square of
 * the wavelength.
 *
 * Lines beginning with a hash symbol (#) are treated as comments and therefore
 * ignored.
 *
 * @param[out] sky       Pointer to sky model structure to fill.
 * @param[in]  filename  Path to the a source list file.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_sky_load_gsm(oskar_Sky* sky, const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_LOAD_GSM_H_ */
