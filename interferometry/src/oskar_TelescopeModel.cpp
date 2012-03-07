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

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_telescope_model_analyse.h"
#include "interferometry/oskar_telescope_model_copy.h"
#include "interferometry/oskar_telescope_model_free.h"
#include "interferometry/oskar_telescope_model_init.h"
#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_multiply_by_wavenumber.h"
#include "interferometry/oskar_telescope_model_resize.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "station/oskar_station_model_load.h"
#include <cstdlib>

oskar_TelescopeModel::oskar_TelescopeModel(int type, int location,
        int n_stations)
{
    if (oskar_telescope_model_init(this, type, location, n_stations))
        throw "Error in oskar_telescope_model_init.";
}

oskar_TelescopeModel::oskar_TelescopeModel(const oskar_TelescopeModel* other,
        int location)
{
    if (oskar_telescope_model_init(this, other->station_x.type, location,
            other->num_stations))
        throw "Error in oskar_telescope_model_init.";
    if (oskar_telescope_model_copy(this, other)) // Copy other to this.
        throw "Error in oskar_telescope_model_copy.";
}

oskar_TelescopeModel::~oskar_TelescopeModel()
{
    if (oskar_telescope_model_free(this))
        throw "Error in oskar_telescope_model_free.";
}

void oskar_TelescopeModel::analyse()
{
    oskar_telescope_model_analyse(this);
}

int oskar_TelescopeModel::load_station_coords(const char* filename,
        double longitude, double latitude, double altitude)
{
    return oskar_telescope_model_load_station_coords(this, filename,
            longitude, latitude, altitude);
}

int oskar_TelescopeModel::location() const
{
    return oskar_telescope_model_location(this);
}

int oskar_TelescopeModel::load_station(int index, const char* filename)
{
    if (index >= this->num_stations)
        return OSKAR_ERR_OUT_OF_RANGE;
    return oskar_station_model_load(&(this->station[index]), filename);
}

int oskar_TelescopeModel::multiply_by_wavenumber(double frequency_hz)
{
    return oskar_telescope_model_multiply_by_wavenumber(this, frequency_hz);
}

int oskar_TelescopeModel::resize(int n_stations)
{
    return oskar_telescope_model_resize(this, n_stations);
}

int oskar_TelescopeModel::type() const
{
    return oskar_telescope_model_type(this);
}
