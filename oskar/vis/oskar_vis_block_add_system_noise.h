/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
 * @param[in,out] vis             Visibility block to which to add noise.
 * @param[in]     header          Visibility header.
 * @param[in]     telescope       Telescope model in use.
 * @param[in,out] station_work    Work buffer of length num_stations.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_vis_block_add_system_noise(oskar_VisBlock* vis,
        const oskar_VisHeader* header, const oskar_Telescope* telescope,
        oskar_Mem* station_work, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
