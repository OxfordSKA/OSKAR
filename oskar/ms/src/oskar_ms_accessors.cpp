/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

using namespace casa;

size_t oskar_ms_column_element_size(const oskar_MeasurementSet* p,
        const char* column)
{
    if (!p->ms || !p->ms->tableDesc().isColumn(column)) return 0;

    switch (p->ms->tableDesc().columnDesc(column).dataType())
    {
    case TpBool:     return sizeof(Bool);
    case TpChar:     return sizeof(Char);
    case TpUChar:    return sizeof(uChar);
    case TpShort:    return sizeof(Short);
    case TpUShort:   return sizeof(uShort);
    case TpInt:      return sizeof(Int);
    case TpUInt:     return sizeof(uInt);
    case TpFloat:    return sizeof(Float);
    case TpDouble:   return sizeof(Double);
    case TpComplex:  return sizeof(Complex);
    case TpDComplex: return sizeof(DComplex);
    default:         return 0;
    }
    return 0;
}

int oskar_ms_column_element_type(const oskar_MeasurementSet* p,
        const char* column)
{
    if (!p->ms || !p->ms->tableDesc().isColumn(column))
        return OSKAR_MS_UNKNOWN_TYPE;

    switch (p->ms->tableDesc().columnDesc(column).dataType())
    {
    case TpBool:     return OSKAR_MS_BOOL;
    case TpChar:     return OSKAR_MS_CHAR;
    case TpUChar:    return OSKAR_MS_UCHAR;
    case TpShort:    return OSKAR_MS_SHORT;
    case TpUShort:   return OSKAR_MS_USHORT;
    case TpInt:      return OSKAR_MS_INT;
    case TpUInt:     return OSKAR_MS_UINT;
    case TpFloat:    return OSKAR_MS_FLOAT;
    case TpDouble:   return OSKAR_MS_DOUBLE;
    case TpComplex:  return OSKAR_MS_COMPLEX;
    case TpDComplex: return OSKAR_MS_DCOMPLEX;
    default:         return OSKAR_MS_UNKNOWN_TYPE;
    }
    return OSKAR_MS_UNKNOWN_TYPE;
}

size_t* oskar_ms_column_shape(const oskar_MeasurementSet* p, const char* column,
        size_t* ndim)
{
    size_t i = 0, *t = 0;
    if (!p->ms || !p->ms->tableDesc().isColumn(column)) return 0;

    const ColumnDesc& cdesc = p->ms->tableDesc().columnDesc(column);
    const IPosition& shape = cdesc.shape();
    if (shape.size() > 0)
    {
        *ndim = (int) shape.size();
        t = (size_t*) calloc(*ndim, sizeof(size_t));
        for (i = 0; i < *ndim; ++i) t[i] = shape(i);
    }
    else if (p->ms->nrow() > 0)
    {
        // If shape is not fixed, return shape of first cell instead.
        TableColumn tc(*(p->ms), column);
        IPosition shape = tc.shape(0);
        if (shape.size() > 0)
        {
            *ndim = (int) shape.size();
            t = (size_t*) calloc(*ndim, sizeof(size_t));
            for (i = 0; i < *ndim; ++i) t[i] = shape(i);
        }
    }
    return t;
}

void oskar_ms_ensure_num_rows(oskar_MeasurementSet* p, unsigned int num)
{
    if (!p->ms) return;
    int rows_to_add = (int)num - (int)(p->ms->nrow());
    if (rows_to_add > 0)
        p->ms->addRow((unsigned int)rows_to_add);
}

double oskar_ms_freq_inc_hz(const oskar_MeasurementSet* p)
{
    return p->freq_inc_hz;
}

double oskar_ms_freq_start_hz(const oskar_MeasurementSet* p)
{
    return p->freq_start_hz;
}

unsigned int oskar_ms_num_channels(const oskar_MeasurementSet* p)
{
    return p->num_channels;
}

unsigned int oskar_ms_num_pols(const oskar_MeasurementSet* p)
{
    return p->num_pols;
}

unsigned int oskar_ms_num_rows(const oskar_MeasurementSet* p)
{
    if (!p->ms) return 0;
    return p->ms->nrow();
}

unsigned int oskar_ms_num_stations(const oskar_MeasurementSet* p)
{
    return p->num_stations;
}

double oskar_ms_phase_centre_ra_rad(const oskar_MeasurementSet* p)
{
    return p->phase_centre_ra;
}

double oskar_ms_phase_centre_dec_rad(const oskar_MeasurementSet* p)
{
    return p->phase_centre_dec;
}

void oskar_ms_set_phase_centre(oskar_MeasurementSet* p, int coord_type,
        double longitude_rad, double latitude_rad)
{
    if (!p->ms || !p->msc) return;

    // Set up the field info.
    MVDirection radec(Quantity(longitude_rad, "rad"),
            Quantity(latitude_rad, "rad"));
    Vector<MDirection> direction(1);
    (void) coord_type;
    direction(0) = MDirection(radec, MDirection::J2000);

    // Write data to the last row of the FIELD table.
    int row = p->ms->field().nrow() - 1;
    if (row < 0)
    {
        p->ms->field().addRow();
        row = 0;
    }
    p->msc->field().delayDirMeasCol().put((unsigned int)row, direction);
    p->msc->field().phaseDirMeasCol().put((unsigned int)row, direction);
    p->msc->field().referenceDirMeasCol().put((unsigned int)row, direction);
    p->phase_centre_ra = longitude_rad;
    p->phase_centre_dec = latitude_rad;
}

template <typename T>
void oskar_ms_set_station_coords(oskar_MeasurementSet* p,
        unsigned int num_stations, const T* x, const T* y, const T* z)
{
    if (!p->ms || !p->msc) return;
    if (num_stations != p->num_stations) return;

    Vector<Double> pos(3, 0.0);
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        pos(0) = x[a]; pos(1) = y[a]; pos(2) = z[a];
        p->msc->antenna().position().put(a, pos);
    }
}

void oskar_ms_set_station_coords_d(oskar_MeasurementSet* p,
        unsigned int num_stations, const double* x, const double* y,
        const double* z)
{
    oskar_ms_set_station_coords(p, num_stations, x, y, z);
}

void oskar_ms_set_station_coords_f(oskar_MeasurementSet* p,
        unsigned int num_stations, const float* x, const float* y,
        const float* z)
{
    oskar_ms_set_station_coords(p, num_stations, x, y, z);
}

void oskar_ms_set_time_range(oskar_MeasurementSet* p)
{
    if (!p->msc) return;

    // Get the old time range.
    Vector<Double> old_range(2, 0.0);
    p->msc->observation().timeRange().get(0, old_range);

    // Compute the new time range.
    Vector<Double> range(2, 0.0);
    range[0] = (old_range[0] <= 0.0 || p->start_time < old_range[0]) ?
            p->start_time : old_range[0];
    range[1] = (p->end_time > old_range[1]) ? p->end_time : old_range[1];
    double release_date = range[1] + 365.25 * 86400.0;

    // Fill observation columns.
    p->msc->observation().timeRange().put(0, range);
    p->msc->observation().releaseDate().put(0, release_date);
}

double oskar_ms_time_inc_sec(const oskar_MeasurementSet* p)
{
    return p->time_inc_sec;
}

double oskar_ms_time_start_mjd_utc(const oskar_MeasurementSet* p)
{
    return p->start_time / 86400.0;
}

