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

#ifndef OSKAR_STATION_H_
#define OSKAR_STATION_H_

/**
 * @file oskar_station.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Station;
#ifndef OSKAR_STATION_TYPEDEF_
#define OSKAR_STATION_TYPEDEF_
typedef struct oskar_Station oskar_Station;
#endif /* OSKAR_STATION_TYPEDEF_ */

enum OSKAR_STATION_TYPE
{
    OSKAR_STATION_TYPE_AA,
    OSKAR_STATION_TYPE_ISOTROPIC,
    OSKAR_STATION_TYPE_GAUSSIAN_BEAM,
    OSKAR_STATION_TYPE_VLA_PBCOR
};

#ifdef __cplusplus
}
#endif

#include <oskar_station_accessors.h>
#include <oskar_station_analyse.h>
#include <oskar_station_create_child_stations.h>
#include <oskar_station_create_copy.h>
#include <oskar_station_create.h>
#include <oskar_station_different.h>
#include <oskar_station_duplicate_first_child.h>
#include <oskar_station_free.h>
#include <oskar_station_load_apodisation.h>
#include <oskar_station_load_element_types.h>
#include <oskar_station_load_feed_angle.h>
#include <oskar_station_load_gain_phase.h>
#include <oskar_station_load_layout.h>
#include <oskar_station_load_mount_types.h>
#include <oskar_station_load_permitted_beams.h>
#include <oskar_station_override_element_feed_angle.h>
#include <oskar_station_override_element_gains.h>
#include <oskar_station_override_element_phases.h>
#include <oskar_station_override_element_time_variable_gains.h>
#include <oskar_station_override_element_time_variable_phases.h>
#include <oskar_station_override_element_xy_position_errors.h>
#include <oskar_station_resize.h>
#include <oskar_station_resize_element_types.h>
#include <oskar_station_save_apodisation.h>
#include <oskar_station_save_element_types.h>
#include <oskar_station_save_feed_angle.h>
#include <oskar_station_save_gain_phase.h>
#include <oskar_station_save_layout.h>
#include <oskar_station_save_mount_types.h>
#include <oskar_station_save_permitted_beams.h>
#include <oskar_station_set_element_coords.h>
#include <oskar_station_set_element_errors.h>
#include <oskar_station_set_element_feed_angle.h>
#include <oskar_station_set_element_mount_type.h>
#include <oskar_station_set_element_type.h>
#include <oskar_station_set_element_weight.h>

#endif /* OSKAR_STATION_H_ */
