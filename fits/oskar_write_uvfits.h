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

#ifndef OSKAR_WRITE_UVFITS_H_
#define OSKAR_WRITE_UVFITS_H_

/**
 * @file oskar_write_uvfits.h
 */

#include "oskar_windows.h"
#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif


DllExport
void oskar_open_uvfits_file(const char* filename, fitsfile** fits_file);

DllExport
void oskar_check_fits_status(const int status, const char* message);


DllExport
void oskar_close_uvfits_file(fitsfile* fits_file);

DllExport
void oskar_write_groups_header(fitsfile* fits_file,
        const long long num_vis, const long num_stokes, const long num_freqs,
        const long num_ra, const long num_dec);



/**
 * @class UVFitsWriter
 *
 * @brief Class to write UVFITS data files.
 *
 * @details
 * see:
 *  http://archive.stsci.edu/fits/fits_standard/fits_standard.html
 *  ftp://ftp.aoc.nrao.edu/pub/software/aips/TEXT/PUBL/FITS-IDI.pdf
 */

//class UVFitsWriter
//{
//    public:
//        UVFitsWriter();
//
//        ~UVFitsWriter();
//
//    public:
//        void open_file(const QString & filename, const bool replace = true);
//
//        void close_file();
//
//        void write_header(const long long num_vis);
//
//        void write_groups_header(const long num_stokes = 1,
//                const long num_freqs = 1, const long num_ra = 1,
//                const long num_dec = 1);
//
//        void write_axis_header(const int id, const QString & ctype,
//                const QString & comment, const double crval, const double cdelt,
//                const double crpix, const double crota);
//
//        void write_param_header(const int id, const QString & type,
//                const QString & comment, const double scale, const double zero);
//
//        int num_amps_per_group();
//
//        void write_data(const float * u, const float * v, const float * w,
//                const float * date, const float * baseline, const float * re,
//                const float * im, const float * weight);
//
//    private:
//        void check_status(const QString & message = QString());
//
//    private:
//        fitsfile * _fptr;       /// CFITSIO structure holding file info.
//        QString _filename;      /// Filename of the open fits file.
//        int _status;            /// CFITSIO error status.
//        int _decimals;          /// Number of decimal places for double keywords.
//        int _num_axis;          /// Number of data axes.
//        QVector<long> _axis_dim;/// Data axis dimensions.
//        int _num_param;         /// Number of parameters per visibility (group)
//        int _num_vis;           /// Number of visibilities (= number of groups)
//};
//

#ifdef __cplusplus
}
#endif

#endif // OSKAR_WRITE_UV_FITS
