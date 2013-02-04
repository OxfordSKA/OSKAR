/*
 * Copyright (c) 2013, The University of Oxford
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

#include "interferometry/oskar_telescope_model_analyse.h"
#include "station/oskar_station_model_analyse.h"
#include "station/oskar_station_model_different.h"

#ifdef __cplusplus
extern "C" {
#endif

static void max_station_size(oskar_StationModel* s, int* max_elements)
{
    int i = 0;
    *max_elements = (s->num_elements > *max_elements) ?
            s->num_elements : *max_elements;
    if (s->child)
    {
        for (i = 0; i < s->num_elements; ++i)
        {
            max_station_size(&s->child[i], max_elements);
        }
    }
}

void oskar_telescope_model_analyse(oskar_TelescopeModel* model, int* status)
{
    int i, finished_identical_station_check, num_stations;

    /* Check all inputs. */
    if (!model || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set default flags. */
    model->identical_stations = 1;
    finished_identical_station_check = 0;

    /* Recursively find the maximum number of elements in any station. */
    num_stations = model->num_stations;
    model->max_station_size = 0;
    for (i = 0; i < num_stations; ++i)
    {
        max_station_size(&model->station[i], &model->max_station_size);
    }

    /* Recursively analyse each station. */
    for (i = 0; i < num_stations; ++i)
    {
        oskar_station_model_analyse(&model->station[i],
                &finished_identical_station_check, status);
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if we need to check every station. */
    if (finished_identical_station_check)
    {
        model->identical_stations = 0;
    }
    else
    {
        /* Check if the stations are different. */
        for (i = 1; i < num_stations; ++i)
        {
            if (oskar_station_model_different(&model->station[0],
                    &model->station[i], status))
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
