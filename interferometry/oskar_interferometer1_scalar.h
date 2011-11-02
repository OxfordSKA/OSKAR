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

#ifndef OSKAR_CUDA_INTERFEROMETER1_SCALAR_H_
#define OSKAR_CUDA_INTERFEROMETER1_SCALAR_H_

/**
 * @file oskar_cuda_interferometer1_scalar.h
 */

#include "oskar_global.h"

#include "sky/oskar_SkyModel.h"
#include "station/oskar_StationModel.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_vector_types.h"
#include "interferometry/oskar_VisData.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief
 * Function to evaluate the interferometer response of a number of
 * aperture array stations to a point source sky model consisting of sources
 * specified by a scalar brightness.
 *
 * @details
 * The telescope geometry, described by the oskar_TelescopeModel structure
 * is expected to be ITRS co-ordinates in units of metres. The function
 * oskar_horizon_plane_to_itrs() can be used to convert from horizon plane
 * coordinates to the ITRS system.
 *
 * Station aperture array geometry, described by the oskar_StationModel
 * structure array must also be specified in metres.
 *
 * See the OSKAR memo 1 for a more detailed description of the co-ordinate
 * system used.
 *
 * @param[in]  telescope         Telescope model structure with station
 *                               co-ordinates in ITRS coordinates, in metres.
 * @param[in]  stations          Array of station model structures containing
 *                               array coordinates, in metres.
 * @param[in]  sky               Global sky model structure.
 * @param[in]  ra0_rad           RA of the pointing phase centre, in radians.
 * @param[in]  dec0_rad          Declination of the pointing phase centre,
 *                               in radians.
 * @param[in]  obs_start_mjd_utc Start date of the observation, in modified
 *                               Julian days (UTC).
 * @param[in]  obs_length_days   Observation length in days.
 * @param[in]  num_vis_dumps     Number of visibility dumps (made by the
 *                               correlator) to make during the observation time.
 * @param[in]  num_vis_ave       Number of averages of the full visibility
 *                               evaluation per visibility dump. Both
 *                               the interferometer phase and beam-pattern
 *                               (E-Jones) is updated for each evaluation.
 * @param[in]  num_fringe_ave    Number of averages per full visibility average
 *                               where only the interferometer phase is updated
 *                               (with a fixed beam-pattern / E-Jones)
 * @param[in]  frequency         Observation frequency, in Hz.
 * @param[in]  bandwidth         Observation channel bandwidth, in Hz.
 * @param[in]  disable_e_jones   Disable evaluation of station beam (i.e. set to 1)
 * @param[out] vis               Structure holding visibility amplitudes and
 *                               baseline co-ordinates.
 *
 * @return CUDA error code.
 */
OSKAR_EXPORT
int oskar_interferometer1_scalar_d(
        const oskar_TelescopeModel_d telescope,
        const oskar_StationModel_d * stations,
        oskar_SkyModelGlobal_d sky,
        const double ra0_rad,
        const double dec0_rad,
        const double obs_start_mjd_utc,
        const double obs_length_days,
        const unsigned num_vis_dumps,
        const unsigned num_vis_ave,
        const unsigned num_fringe_ave,
        const double frequency,
        const double bandwidth,
        const bool disable_e_jones,
        oskar_VisData_d* vis
);




/**
 * @brief
 * Single precision implementation of above function.
 */
OSKAR_EXPORT
int oskar_interferometer1_scalar_f(
        const oskar_TelescopeModel_f telescope,
        const oskar_StationModel_f * stations,
        oskar_SkyModelGlobal_f sky,
        const float ra0_rad,
        const float dec0_rad,
        const float obs_start_mjd_utc,
        const float obs_length_days,
        const unsigned num_vis_dumps,
        const unsigned num_vis_ave,
        const unsigned num_fringe_ave,
        const float frequency,
        const float bandwidth,
        const bool disable_e_jones,
        oskar_VisData_f* vis
);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CUDA_INTERFEROMETER1_SCALAR_H_ */
