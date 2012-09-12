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

#ifndef OSKAR_INTERFEROMETER_SCALAR_H_
#define OSKAR_INTERFEROMETER_SCALAR_H_

/**
 * @file oskar_interferometer_scalar.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_Settings.h"
#include "interferometry/oskar_Visibilities.h"
#include "sky/oskar_SkyModel.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Main interferometer simulation function (scalar version).
 *
 * @details
 * This function produces simulated visibilities from an interferometer.
 *
 * @param[out]    vis_amp        Output visibility amplitudes.
 * @param[in,out] log            Pointer to log structure to use.
 * @param[in]     sky            Sky model structure.
 * @param[in]     telescope      Telescope model structure.
 * @param[in]     settings       Simulation settings.
 * @param[in]     frequency      Observation frequency in Hz.
 * @param[in]     chunk_index    Sky chunk (index) to be processed.
 * @param[in]     num_sky_chunks Total number of sky chunks.
 */
OSKAR_EXPORT
int oskar_interferometer_scalar(oskar_Mem* vis_amp, oskar_Log* log,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        const oskar_Settings* settings, double frequency, int chunk_index,
        int num_sky_chunks);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_INTERFEROMETER_SCALAR_H_ */
