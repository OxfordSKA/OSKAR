/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_MODEL_NOISE_LOAD_H_
#define OSKAR_TELESCOPE_MODEL_NOISE_LOAD_H_

/**
 * @file oskar_telescope_model_noise_load.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_Log.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Populates the oskar_SystemNoiseModel structures of stations in the telescope
 * model.
 *
 * @details
 * Loads and/or evaluates lookup tables of RMS noise v.s. frequency for
 * each station in the telescope model.
 *
 * Noise values are derived from settings in the oskar_Settings structure
 * either directly or by reading files in the telescope model directory.
 *
 * The telescope model being populated must have already been allocated in
 * CPU memory.
 *
 * Note that oskar_telescope_model_load_config() should have been called first
 * to allocate the telescope model for the correct number of stations and/or
 * child elements.
 *
 * @param[out]    telescope Telescope structure pointer.
 * @param[in,out] log       Pointer to log structure.
 * @param[in]     settings  Pointer to settings structure.
 * @param[in,out] status    Status return code.
 */
OSKAR_APPS_EXPORT
void oskar_telescope_model_noise_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_Settings* settings, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_MODEL_NOISE_LOAD_H_ */
