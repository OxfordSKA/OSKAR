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

#ifndef OSKAR_SKY_MODEL_LOAD_H_
#define OSKAR_SKY_MODEL_LOAD_H_

/**
 * @file oskar_sky_model_load.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"


/**
 * @brief
 * Loads sources from a plain text source file into an OSKAR sky model
 * structure.
 *
 * @details
 * Source files are plain ASCII files consisting of the following 8 columns:
 * - RA (deg),
 * - Dec (deg),
 * - Stokes I (Jy),
 * - Stokes Q (Jy),
 * - Stokes U (Jy),
 * - Stokes V (Jy),
 * - Reference frequency (Hz),
 * - Spectral index
 *
 * Columns 4 to 8 (Q, U, V, Reference frequency and Spectral index) are optional
 * and defaulted to zero if omitted.
 *
 * The columns must be space or comma separated.
 *
 * Lines beginning with a hash symbol (#) are treated as comments and therefore
 * ignored.
 *
 * @param[in]  filename  Path to the a source list file.
 * @param[out] sky       Pointer to sky model structure to fill.
 */
#ifdef __cplusplus
extern "C"
#endif
OSKAR_EXPORT
int oskar_sky_model_load(const char* filename, oskar_SkyModel* sky);






#ifdef __cplusplus
extern "C" {
#endif

/**
 * DEPRECATED
 * @brief Loads sources from a plain text source file into an oskar global sky
 * model structure (double precision).
 *
 * @details
 * Source files are plain ascii files consisting of the following 8 columns:
 *  RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), reference frequency (Hz), spectral index
 *
 *  Lines beginning with a # symbol are treated as comments and therefore ignored.
 *
 * @param[in]  file_path  Path to the a source list file.
 * @param[out] sky        Pointer to global sky model structure.
 */
OSKAR_EXPORT
void oskar_sky_model_load_d(const char* file_path, oskar_SkyModelGlobal_d* sky);



/**
 * DEPRECATED
 * @brief Loads sources from a plain text source file into an oskar global sky
 * model structure (single precision).
 *
 * @details
 * Source files are plain ascii files consisting of the following 8 columns:
 *  RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), reference frequency (Hz), spectral index
 *
 *  Lines beginning with a # symbol are treated as comments and therefore ignored.
 *
 * @param[in]  file_path  Path to the a source list file.
 * @param[out] sky        Pointer to global sky model structure.
 */
OSKAR_EXPORT
void oskar_sky_model_load_f(const char* file_path, oskar_SkyModelGlobal_f* sky);



#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_MODEL_LOAD_H_ */
