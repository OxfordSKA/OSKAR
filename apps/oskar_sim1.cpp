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


#include "apps/lib/oskar_Settings.h"
#include <cstdio>
#include <cstdlib>
#include "interferometry/oskar_evaluate_jones_K.h"
#include "interferometry/oskar_correlate.h"
#include "station/oskar_evaluate_jones_E.h"
#include "math/oskar_jones_join.h"

int main(int argc, char** argv)
{
    oskar_Jones* E;
    oskar_Jones* K;
    oskar_Jones* J;
    oskar_Sky* sky;
    oskar_Telescope* telescope;
    oskar_Visibilities* vis;

    oskar_load_stations_layouts(telescope, "station_directory");
    oskar_load_station_positions(telescope, "telescope_layout_file");
    oskar_load_global_sky_model(sky, "sky_model_file");

    // initialise E, J, K etc.
    //...

    for (int j = 0; j < num_vis_dumps; ++j)
    {
        for (int i = 0; i < num_vis_ave; ++i)
        {
            double last = 0.0;
            oskar_evaluate_jones_E(E, sky, telescope, last);

            for (int k = 0; k < num_fringe_ave; ++k)
            {
                last = 0.0 + 0.1;
                oskar_evaluate_jones_K(K, sky, telescope, last);
                oskar_jones_join(J, K, E);
                oskar_correlate(vis, J, telescope, sky, last);
            }
        }
        // Dump vis to MS?
    }

    return EXIT_SUCCESS;
}


