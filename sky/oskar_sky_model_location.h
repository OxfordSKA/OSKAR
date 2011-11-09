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

#ifndef OSKAR_SKY_MODEL_LOCATION_H_
#define OSKAR_SKY_MODEL_LOCATION_H_

/**
 * @file oskar_sky_model_location.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Checks if the OSKAR sky model memory is at the specified location.
 *
 * @details
 * \p location should be an oskar_Mem data location ID.
 *
 * If the location is found to be inconsistent between all of the oskar_Mem
 * structures held in the sky model the location check is considered false.
 *
 * @param sky       Pointer to sky model structure.
 * @param location  oskar_Mem location type to check against.

 * @return 1 (true) if the sky model is of the specified location, 0 otherwise.
 */
int oskar_sky_model_is_location(const oskar_SkyModel* sky, int location);

/**
 * @brief Returns the oskar_Mem location ID for the sky structure or an
 * error code if an invalid location is found.
 *
 * @param sky Pointer to an OSKAR sky model structure.
 *
 * @return oskar_Mem data location or error code.
 */
int oskar_sky_model_location(const oskar_SkyModel* sky);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_MODEL_LOCATION_H_ */
