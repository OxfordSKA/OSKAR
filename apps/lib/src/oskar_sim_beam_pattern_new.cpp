/*
 * Copyright (c) 2014, The University of Oxford
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

#include <cuda_runtime_api.h>

#include <oskar_settings_load.h>
#include <oskar_settings_log.h>

#include <oskar_sim_beam_pattern.h>
#include <oskar_set_up_telescope.h>
#include <oskar_beam_pattern_generate_coordinates.h>

#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_evaluate_average_cross_power_beam.h>
#include <oskar_evaluate_station_beam.h>
#include <oskar_evaluate_jones_E.h>
#include <oskar_jones.h>
#include <oskar_log.h>
#include <oskar_random_state.h>
#include <oskar_station_work.h>
#include <oskar_telescope.h>
#include <oskar_timer.h>

#include <oskar_settings_free.h>

#include <oskar_cmath.h>
#include <cstring>
#include <vector>

using std::vector;

#include <fitsio.h>
struct oskar_PixelDataHandle
{
    int num_dims;
    int* dim;

    int num_reorder_groups;
    int num_handles;
    int* data_type;
    fitsfile** handle_fits;
    FILE** handle_ascii;
};

/* TODO Write new version of oskar_sim_beam_pattern that doesn't use
 * oskar_Image, and evaluates pixels in chunks. */
