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

#include "oskar_global.h"

#include "station/oskar_evaluate_jones_E.h"

#include "sky/oskar_SkyModel.h"
#include "sky/oskar_ra_dec_to_hor_lmn.h"
#include "sky/oskar_cuda_ra_dec_to_hor_lmn.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "math/oskar_Jones.h"
#include "station/oskar_evaluate_station_beam.h"


#ifdef __cplusplus
extern "C"
#endif
int oskar_evaluate_jones_E(oskar_Jones* E, oskar_SkyModel* sky,
        oskar_TelescopeModel* telescope, double gast)
{
    // Consistency and validation checks on input arguments.
    // -------------------------------------------------------------------------
    if (E == NULL || sky == NULL || telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (telescope->num_stations == 0)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if (telescope->station == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // TODO input argument validation
    // - Check the sky, telescope and Jones for type consistency.
    //    - double / single precision consistency
    //    - format of E (scalar, matrix) with presence of element pattern data.
    // - Work out what checking has to be done here and what can be delayed to
    //   next level wrapper functions.






    // Evaluate the station beam for each station for each source position.
    // ------------------------------------------------------------------------

    // TODO
    // - function to evaluate the station beam values for one station.
    //

    if (telescope->identical_stations && telescope->use_common_sky)
    {
            // Evaluate horizontal l,m,n once and the station beam for
            // station 0 and copy the result into the data for other stations
            // in E.

            const oskar_StationModel* station0 = &telescope->station[0];

            // Evaluate the horizontal l,m,m coordinates for beam phase centre
            // and sources.
//            double beam_hor_l, beam_hor_m, beam_hor_n;
//            oskar_ra_dec_to_hor_lmn_d(1, &station0->ra0, &station0->dec0,
//                    gast, station0->latitude, &beam_hor_l, &beam_hor_m,
//                    &beam_hor_n);
//
            // sources above horizon?
            //oskar_cuda_ra_dec_to_hor_lmn_d(m, ra, dec, lst, lat, hor_l, hor_m, hor_n);
            // TODO --> oskar_evalute_beam_hor_lmn(station0, gast);
            // TODO --> oskar_evalute_source_hor_lmn(sky, station0, gast);

            // Evaluate station beam.
            // NOTE E and station index could be replaced by a
            // oskar_Jones_row structure which holds a pointer to the row
            // for the station in the oskar_Jones along with its type location
            // and dimensions. e.g.
            // oskar_Jones_row station0_E = E.get_row(station_index)
            // oskar_evalute_station_beam(station0_E, sky, station0)
//            oskar_evalute_station_beam(E, 0, sky, station0);
    }
    else
    {
        if (telescope->use_common_sky)
        {
            // Evaluate horizontal l,m,n once and use it for evaluating
            // the station beam for each station.
            // loop over stations to evaluate E.
            for (int i = 0; i < telescope->num_stations; ++i)
            {
                const oskar_StationModel* station = &telescope->station[i];

            }
        }
        else
        {
            // Evaluate horizontal l,m,n and the station beam for each station.
            // loop over stations to evaluate E.
            for (int i = 0; i < telescope->num_stations; ++i)
            {
                const oskar_StationModel* station = &telescope->station[i];

            }
        }
    }
    return 0;
}

