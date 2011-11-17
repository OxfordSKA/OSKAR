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


#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_append.h"
#include "sky/oskar_sky_model_compute_relative_lmn.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_resize.h"
#include "sky/oskar_sky_model_scale_by_spectral_index.h"
#include "sky/oskar_sky_model_set_source.h"
#include "sky/oskar_sky_model_type.h"

oskar_SkyModel::oskar_SkyModel(int type, int location, int num_sources)
: num_sources(num_sources),
  RA(type, location, num_sources),
  Dec(type, location, num_sources),
  I(type, location, num_sources),
  Q(type, location, num_sources),
  U(type, location, num_sources),
  V(type, location, num_sources),
  reference_freq(type, location, num_sources),
  spectral_index(type, location, num_sources),
  rel_l(type, location, num_sources),
  rel_m(type, location, num_sources),
  rel_n(type, location, num_sources)
{
}

oskar_SkyModel::oskar_SkyModel(const oskar_SkyModel* other, int location)
: num_sources(other->num_sources),
  RA(&other->RA, location),
  Dec(&other->Dec, location),
  I(&other->I, location),
  Q(&other->Q, location),
  U(&other->U, location),
  V(&other->V, location),
  reference_freq(&other->reference_freq, location),
  spectral_index(&other->spectral_index, location),
  rel_l(&other->rel_l, location),
  rel_m(&other->rel_m, location),
  rel_n(&other->rel_n, location)
{
}

oskar_SkyModel::oskar_SkyModel(const char* filename, int type, int location)
: num_sources(0),
  RA(type, location, 0),
  Dec(type, location, 0),
  I(type, location, 0),
  Q(type, location, 0),
  U(type, location, 0),
  V(type, location, 0),
  reference_freq(type, location, 0),
  spectral_index(type, location, 0),
  rel_l(type, location, 0),
  rel_m(type, location, 0),
  rel_n(type, location, 0)
{
    if (oskar_sky_model_load(filename, this) != 0)
        throw "Error in oskar_sky_model_load";
}

oskar_SkyModel::~oskar_SkyModel()
{
}

int oskar_SkyModel::append(const oskar_SkyModel* other)
{
    return oskar_sky_model_append(this, other);
}

int oskar_SkyModel::compute_relative_lmn(double ra0, double dec0)
{
    return oskar_sky_model_compute_relative_lmn(this, ra0, dec0);
}

int oskar_SkyModel::load(const char* filename)
{
    return oskar_sky_model_load(filename, this);
}

int oskar_SkyModel::resize(int num_sources)
{
    return oskar_sky_model_resize(this, num_sources);
}

int oskar_SkyModel::scale_by_spectral_index(double frequency)
{
    return oskar_sky_model_scale_by_spectral_index(this, frequency);
}

int oskar_SkyModel::set_source(int index, double ra, double dec, double I,
        double Q, double U, double V, double ref_frequency,
        double spectral_index)
{
    return oskar_sky_model_set_source(this, index, ra, dec, I, Q, U, V,
            ref_frequency, spectral_index);
}

int oskar_SkyModel::type() const
{
    return oskar_sky_model_type(this);
}

int oskar_SkyModel::location() const
{
    return oskar_sky_model_location(this);
}

bool oskar_SkyModel::is_double() const
{
    return (oskar_sky_model_type(this) == OSKAR_DOUBLE);
}
