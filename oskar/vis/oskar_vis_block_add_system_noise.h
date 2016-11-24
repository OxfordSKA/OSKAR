/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#ifndef OSKAR_VIS_BLOCK_ADD_SYSTEM_NOISE_H_
#define OSKAR_VIS_BLOCK_ADD_SYSTEM_NOISE_H_

/**
 * @file oskar_vis_block_add_system_noise.h
 */

#include <oskar_global.h>
#include <telescope/oskar_telescope.h>
#include <vis/oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Add a random Gaussian noise component to the visibilities.
 *
 * @param[in,out] vis             Visibility structure to which to add noise.
 * @param[in]     telescope       Telescope model in use.
 * @param[in]     block_index     Simulation time index for the block.
 * @param[in,out] station_work    Work buffer of length num_stations.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_vis_block_add_system_noise(oskar_VisBlock* vis,
        const oskar_VisHeader* header, const oskar_Telescope* telescope,
        unsigned int block_index, oskar_Mem* station_work, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_BLOCK_ADD_SYSTEM_NOISE_H_ */
