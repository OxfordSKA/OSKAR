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


#ifndef OSKAR_SKY_MODEL_CHECK_MEM_H_
#define OSKAR_SKY_MODEL_CHECK_MEM_H_

/**
 * @file oskar_sky_model_check_mem.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"
#include "utility/oskar_Mem.h"
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Checks if the OSKAR sky model is of the specified type.
 *
 * @details
 * \p type should be an oskar_Mem data type.
 *
 * If the types are found to be inconsistent between all of the oskar_Mem
 * structures held in the sky model the type check is considered false.
 *
 * @param sky   Pointer to sky model structure.
 * @param type  oskar_Mem data type to check against.

 * @return 1 (true) if the sky model is of the specified type, 0 otherwise.
 */
inline int oskar_sky_model_is_type(const oskar_SkyModel* sky, const int type)
{
    return (sky->RA.type() == type &&
            sky->Dec.type() == type &&
            sky->I.type() == type &&
            sky->Q.type() == type &&
            sky->U.type() == type &&
            sky->V.type() == type &&
            sky->reference_freq.type() == type &&
            sky->spectral_index.type() == type &&
            sky->rel_l.type() == type &&
            sky->rel_m.type() == type &&
            sky->rel_n.type() == type);
}

/**
 * @brief Returns the oskar_Mem type ID for the sky structure or an error
 * code if an invalid type is found.
 *
 * @param sky Pointer to an OSKAR sky model structure.
 *
 * @return oskar_Mem data type or error code.
 */
inline int oskar_sky_model_type(const oskar_SkyModel* sky)
{
    if (sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_sky_model_is_type(sky, OSKAR_DOUBLE))
        return OSKAR_DOUBLE;
    else if (oskar_sky_model_is_type(sky, OSKAR_SINGLE))
        return OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;
}

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
inline int oskar_sky_model_is_location(const oskar_SkyModel* sky, const int location)
{
    return (sky->RA.location() == location &&
            sky->Dec.location() == location &&
            sky->I.location() == location &&
            sky->Q.location() == location &&
            sky->U.location() == location &&
            sky->V.location() == location &&
            sky->reference_freq.location() == location &&
            sky->spectral_index.location() == location &&
            sky->rel_l.location() == location &&
            sky->rel_m.location() == location &&
            sky->rel_n.location() == location);
}

/**
 * @brief Returns the oskar_Mem location ID for the sky structure or an error
 * code if an invalid location is found.
 *
 * @param sky Pointer to an OSKAR sky model structure.
 *
 * @return oskar_Mem data location or error code.
 */
inline int oskar_sky_model_location(const oskar_SkyModel* sky)
{
    if (sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_sky_model_is_location(sky, OSKAR_LOCATION_CPU))
        return OSKAR_LOCATION_CPU;
    else if (oskar_sky_model_is_location(sky, OSKAR_LOCATION_GPU))
        return OSKAR_LOCATION_GPU;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;
}


#ifdef __cplusplus
}
#endif

#endif // OSKAR_SKY_MODEL_CHECK_MEM_H_
