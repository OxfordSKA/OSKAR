/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

#include <cstdlib>
#include <cstdio>

using namespace casacore;

static oskar_MeasurementSet* _oskar_ms_open(const char* filename, bool readonly)
{
    oskar_MeasurementSet* p = (oskar_MeasurementSet*)
            calloc(1, sizeof(oskar_MeasurementSet));

    try
    {
        // Create the MeasurementSet. Storage managers are recreated as needed.
        TableLock::LockOption lock = TableLock::PermanentLocking;
        Table::TableOption mode = Table::Update;
        if (readonly)
        {
            lock = TableLock::NoLocking;
            mode = Table::Old;
        }
        p->ms = new MeasurementSet(filename, lock, mode);

        // Create the MSMainColumns and MSColumns objects for accessing data
        // in the main table and subtables.
        p->msc = new MSColumns(*(p->ms));
        p->msmc = new MSMainColumns(*(p->ms));
    }
    catch (AipsError& e)
    {
        fprintf(stderr, "Caught AipsError: %s\n", e.what());
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }

    // Refuse to open if there is more than one spectral window.
    if (p->ms->spectralWindow().nrow() != 1)
    {
        fprintf(stderr, "OSKAR can read Measurement Sets with one spectral "
                "window only. Use 'split' or 'mstransform' in CASA to select "
                "the spectral window first.\n");
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }

    // Refuse to open if there is more than one field.
    if (p->ms->field().nrow() != 1)
    {
        fprintf(stderr, "OSKAR can read Measurement Sets with one target "
                "field only. Use 'split' or 'mstransform' in CASA to select "
                "the target field first.\n");
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }

    // Get the data dimensions.
    p->num_pols = 0;
    p->num_channels = 0;
    p->num_receptors = 2;
    if (p->ms->polarization().nrow() > 0)
        p->num_pols = p->msc->polarization().numCorr().get(0);
    if (p->ms->spectralWindow().nrow() > 0)
    {
        p->num_channels = p->msc->spectralWindow().numChan().get(0);
        p->freq_start_hz = p->msc->spectralWindow().refFrequency().get(0);
        p->freq_inc_hz = (p->msc->spectralWindow().chanWidth().get(0))(
                IPosition(1, 0));
    }
    p->num_stations = p->ms->antenna().nrow();
    if (p->ms->nrow() > 0)
        p->time_inc_sec = p->msc->interval().get(0);

    // Get the phase centre.
    p->phase_centre_type = 0;
    p->phase_centre_rad[0] = 0.0;
    p->phase_centre_rad[1] = 0.0;
    if (p->ms->field().nrow() > 0)
    {
        Vector<MDirection> dir;
        p->msc->field().phaseDirMeasCol().get(0, dir, true);
        if (dir.size() > 0)
        {
            Vector<Double> v = dir(0).getAngle().getValue();
            p->phase_centre_rad[0] = v(0);
            p->phase_centre_rad[1] = v(1);
            const String& type = dir(0).tellMe();
            if (type == "J2000")
                p->phase_centre_type = 0;
            else if (type.startsWith("AZEL"))
                p->phase_centre_type = 1;
        }
    }

    // Get the time range.
    Vector<Double> range(2, 0.0);
    if (p->msc->observation().nrow() > 0)
        p->msc->observation().timeRange().get(0, range);
    p->start_time = range[0];
    p->end_time = range[1];

    return p;
}

oskar_MeasurementSet* oskar_ms_open(const char* filename)
{
    return _oskar_ms_open(filename, false);
}

oskar_MeasurementSet* oskar_ms_open_readonly(const char* filename)
{
    return _oskar_ms_open(filename, true);
}
