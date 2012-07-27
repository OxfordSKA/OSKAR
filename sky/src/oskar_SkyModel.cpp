/*
 * Copyright (c) 2012, The University of Oxford
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

#include "sky/oskar_sky_model_all_headers.h"

oskar_SkyModel::oskar_SkyModel(int type, int location, int num_sources)
{
	int err = oskar_sky_model_init(this, type, location, num_sources);
    if (err) throw err;
}

oskar_SkyModel::oskar_SkyModel(const oskar_SkyModel* other, int location)
{
	int err;
	err = oskar_sky_model_init(this, other->type(), location,
			other->num_sources);
    if (err) throw err;
    err = oskar_sky_model_copy(this, other);
    if (err) throw err;
}

oskar_SkyModel::oskar_SkyModel(const char* filename, int type, int location)
{
	int err;
	err = oskar_sky_model_init(this, type, location, 0);
    if (err) throw err;
    err = oskar_sky_model_load(this, filename);
    if (err) throw err;
}

oskar_SkyModel::~oskar_SkyModel()
{
	int err = oskar_sky_model_free(this);
    if (err) throw err;
}

int oskar_SkyModel::append(const oskar_SkyModel* other)
{
    return oskar_sky_model_append(this, other);
}

int oskar_SkyModel::compute_relative_lmn(double ra0, double dec0)
{
    return oskar_sky_model_compute_relative_lmn(this, ra0, dec0);
}

int oskar_SkyModel::copy_to(oskar_SkyModel* other)
{
    return oskar_sky_model_copy(other, this);
}

int oskar_SkyModel::filter_by_flux(double min_I, double max_I)
{
    return oskar_sky_model_filter_by_flux(this, min_I, max_I);
}

int oskar_SkyModel::filter_by_radius(double inner_radius, double outer_radius,
        double ra0, double dec0)
{
    return oskar_sky_model_filter_by_radius(this, inner_radius, outer_radius,
            ra0, dec0);
}

int oskar_SkyModel::load(const char* filename)
{
    return oskar_sky_model_load(this, filename);
}

int oskar_SkyModel::load_gsm(oskar_Log* log, const char* filename)
{
    return oskar_sky_model_load_gsm(this, log, filename);
}

int oskar_SkyModel::location() const
{
    return oskar_sky_model_location(this);
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
        double spectral_index, double FWHM_major, double FWHM_minor,
        double position_angle)
{
    return oskar_sky_model_set_source(this, index, ra, dec, I, Q, U, V,
            ref_frequency, spectral_index, FWHM_major, FWHM_minor,
            position_angle);
}

int oskar_SkyModel::type() const
{
    return oskar_sky_model_type(this);
}

int oskar_SkyModel::write(const char* filename)
{
    return oskar_sky_model_write(filename, this);
}
