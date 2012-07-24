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

#ifndef OSKAR_TELESCOPE_MODEL_CONFIG_LOAD_H_
#define OSKAR_TELESCOPE_MODEL_CONFIG_LOAD_H_

/**
 * @file oskar_telescope_model_config_load.h
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
 * Loads the layout.txt and config.txt files from a telescope model directory
 * populating the relevant parts of an OSKAR telescope model structure.
 *
 * @details
 * The telescope model must be initialised and in CPU memory.
 *
 * @param[out]    telescope  Pointer to empty telescope model structure to fill.
 * @param[in,out] log        Pointer to log structure to use.
 * @param[in]     settings   Pointer to an OSKAR telescope settings structure.
 *
 * @return An OSKAR error code.
 */
OSKAR_EXPORT
int oskar_telescope_model_config_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings);

/**
 * @brief
 * Overrides settings loaded in config.txt files based on values specified
 * in the settings.
 *
 * @details
 * The telescope model must be initialised and in CPU memory.
 *
 * @param[out]    telescope  Pointer to empty telescope model structure to fill.
 * @param[in]     settings   Pointer to and OSKAR telescope settings structure.
 *
 * @return An OSKAR error code.
 */
OSKAR_EXPORT
int oskar_telescope_model_config_override(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_MODEL_CONFIG_LOAD_H_ */
