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

#ifndef OSKAR_SET_UP_SKY_H_
#define OSKAR_SET_UP_SKY_H_

/**
 * @file oskar_set_up_sky.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Settings.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a sky model from the simulation settings.
 *
 * @details
 * This function creates and returns an array of fully populated sky data
 * structures from the given settings object.
 *
 * The data in the structure that is returned resides in CPU memory.
 *
 * @param[out] num_chunks The number of sky model chunks to use.
 * @param[out] sky_chunks Pointer to the array of source chunks.
 * @param[in,out] log  A pointer to the log structure to use.
 * @param[in] settings A pointer to the settings structure.
 */
OSKAR_APPS_EXPORT
int oskar_set_up_sky(int* num_chunks, oskar_SkyModel** sky_chunks,
        oskar_Log* log, const oskar_Settings* settings);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SET_UP_SKY_H_ */
