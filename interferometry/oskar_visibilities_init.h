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

#ifndef OSKAR_VISIBILITIES_INIT_H_
#define OSKAR_VISIBILITIES_INIT_H_

/**
 * @file oskar_visibilities_init.h
 */


#include "oskar_global.h"
#include "interferometry/oskar_Visibilities.h"

/**
 * @brief Initialise the specified oskar_Visibility data structure.
 *
 * @details
 * This function will overwrite any memory currently in the visibility structure.
 *
 * @param vis               Pointer to the visibility data structure to initialise.
 * @param amp_type          OSKAR memory type for the visibility amplitudes.
 * @param location          Memory location (OSKAR_LOCATION_CPU or OSKAR_LOCAITON_GPU).
 * @param num_channels      Number of frequency channels.
 * @param num_times         Number of time samples.
 * @param num_baselines     Number of baselines.
 */
#ifdef __cplusplus
extern "C"
#endif
OSKAR_EXPORT
int oskar_visibilities_init(oskar_Visibilities* vis, int amp_type, int location,
        int num_channels, int num_times, int num_baselines);

#endif /* OSKAR_VISIBILITIES_INIT_H_ */
