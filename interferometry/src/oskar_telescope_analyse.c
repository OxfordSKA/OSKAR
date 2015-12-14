/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <private_telescope.h>
#include <oskar_telescope.h>

#include <oskar_station_analyse.h>
#include <oskar_station_different.h>

#ifdef __cplusplus
extern "C" {
#endif

static void max_station_size_and_depth(const oskar_Station* s,
        int* max_elements, int* max_depth, int depth)
{
    int i = 0, num_elements;
    num_elements = oskar_station_num_elements(s);
    *max_elements = num_elements > *max_elements ? num_elements : *max_elements;
    *max_depth = depth > *max_depth ? depth : *max_depth;
    if (oskar_station_has_child(s))
    {
        for (i = 0; i < num_elements; ++i)
        {
            max_station_size_and_depth(oskar_station_child_const(s, i),
                    max_elements, max_depth, depth + 1);
        }
    }
}

void oskar_telescope_analyse(oskar_Telescope* model, int* status)
{
    int i = 0, finished_identical_station_check = 0, num_stations;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set default flags. */
    model->identical_stations = 1;

    /* Recursively find the maximum number of elements in any station. */
    num_stations = model->num_stations;
    model->max_station_size = 0;
    for (i = 0; i < num_stations; ++i)
    {
        max_station_size_and_depth(oskar_telescope_station_const(model, i),
                &model->max_station_size, &model->max_station_depth, 1);
    }

    /* Recursively analyse each station. */
    for (i = 0; i < num_stations; ++i)
    {
        oskar_station_analyse(oskar_telescope_station(model, i),
                &finished_identical_station_check, status);
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if we need to examine every station. */
    if (finished_identical_station_check)
    {
        model->identical_stations = 0;
    }
    else
    {
        /* Check if the stations are different. */
        for (i = 1; i < num_stations; ++i)
        {
            if (oskar_station_different(oskar_telescope_station(model, 0),
                    oskar_telescope_station(model, i), status))
            {
                model->identical_stations = 0;
                break;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
