/*
 * Copyright (c) 2012, The University of Oxford
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
 * TODO this isnt really just a loader... better function name?
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
 * Populates the noise std.dev. of an OSKAR telescope model.
 *
 * @details
 * Based on the settings and the files in the telescope model directory.
 *
 * @param telescope Telescope structure pointer
 * @param log       Pointer to log structure.
 * @param settings  Pointer to settings structure.
 *
 * @return An OSKAR error code.
 */
OSKAR_APPS_EXPORT
int oskar_telescope_model_noise_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_Settings* settings);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_MODEL_NOISE_LOAD_H_ */
