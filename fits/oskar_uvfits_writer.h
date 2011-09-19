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

#ifndef OSKAR_UVFITS_WRITER_H_
#define OSKAR_UVFITS_WRITER_H_

/**
 * @file oskar_uvfits_writer.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_VisData.h"
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif


struct oskar_uvfits
{
    fitsfile* fptr;
    int       status;
    int       decimals;
    int       num_axes;
    int       num_param;
};
typedef struct oskar_uvfits oskar_uvfits;


enum oskar_uvfits_axis_type
{ GROUPS_NONE = 0, AMP = 1, STOKES = 2, FREQ = 3, RA = 4, DEC = 5 };

struct oskar_uvfits_header
{
    long  num_axes;
    long* axis_dim;

    long long num_param; // == pcount
    long long gcount;    // == num_vis
};




OSKAR_EXPORT
void oskar_uvfits_open(const char* filename, oskar_uvfits* fits);

OSKAR_EXPORT
void oskar_check_fits_status(const int status, const char* message);

OSKAR_EXPORT
void oskar_close_uvfits_file(fitsfile* fits_file);

OSKAR_EXPORT
void oskar_write_groups_header(fitsfile* fits_file, const long long num_vis);

OSKAR_EXPORT
void oskar_write_header(fitsfile* fits_file, const char* filename, double ra0,
        double dec0, double frequency0, double date0);

OSKAR_EXPORT
void oskar_write_axis_header(fitsfile* fits_file, const int id,
        const char* ctype, const char* comment, const double crval,
        const double cdelt, const double crpix, const double crota);

OSKAR_EXPORT
void oskar_write_param_header(fitsfile* fits_file, const int id,
        const char* type, const char* comment, const double scale,
        const double zero);

OSKAR_EXPORT
void oskar_write_data(fitsfile* fits_file, const oskar_VisData_d* vis,
        const double* weight, const double* date, const double* baseline);

OSKAR_EXPORT
int oskar_uvfits_baseline_id(int ant1, int ant2);


#ifdef __cplusplus
}
#endif

#endif // OSKAR_UVFITS_WRITER_H_
