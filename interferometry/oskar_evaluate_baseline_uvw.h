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

#ifndef OSKAR_EVALUATE_BASELINE_UVW_H_
#define OSKAR_EVALUATE_BASELINE_UVW_H_

/**
 * @file oskar_evaluate_baseline_uvw.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_SettingsObservation.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the baseline (u,v,w) coordinates for the simulation.
 *
 * @details
 * This function evaluates the baseline (u,v,w) coordinates from the beam phase
 * centre and station (x,y,z) coordinates in the telescope data structure,
 * and the supplied simulation time parameters.
 *
 * The output coordinates are for the whole observation, so the output arrays
 * must have dimension of (at least) num_baselines * num_vis_dumps.
 *
 * @param[out] uu        Output baseline u coordinates for whole observation.
 * @param[out] vv        Output baseline v coordinates for whole observation.
 * @param[out] ww        Output baseline w coordinates for whole observation.
 * @param[in]  telescope Telescope model structure.
 * @param[in]  obs       Simulation observation settings (used for time data).
 * @param[in,out]  work  Pointer to work buffer to use (>= 3 * num_stations).
 */
OSKAR_EXPORT
int oskar_evaluate_baseline_uvw(oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        const oskar_TelescopeModel* telescope,
        const oskar_SettingsObservation* obs, oskar_Mem* work);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_BASELINE_UVW_H_ */
