/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

#include <cstdlib>
#include <cstdio>

using namespace casacore;

static oskar_MeasurementSet* oskar_ms_open_private(
        const char* filename, bool readonly)
{
    oskar_MeasurementSet* p = (oskar_MeasurementSet*)
            calloc(1, sizeof(oskar_MeasurementSet));
    p->num_receptors = 2;
    Vector<Double> range(2, 0.0);

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
#ifdef OSKAR_MS_NEW
        p->ms = new Table(filename, lock, mode);
        oskar_ms_bind_refs(p);
#else
        p->ms = new MeasurementSet(filename, lock, mode);

        // Create the MSMainColumns and MSColumns objects for accessing data
        // in the main table and subtables.
        p->msc = new MSColumns(*(p->ms));
        p->msmc = new MSMainColumns(*(p->ms));
#endif
    }
    catch (AipsError& e)
    {
        fprintf(stderr, "Caught AipsError: %s\n", e.what());
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }

#ifdef OSKAR_MS_NEW
    Table spw(p->ms->tableName() + "/SPECTRAL_WINDOW", Table::Old);
    Table field(p->ms->tableName() + "/FIELD", Table::Old);
    const int num_spectral_windows = spw.nrow();
    const int num_fields = field.nrow();
#else
    const int num_spectral_windows = p->ms->spectralWindow().nrow();
    const int num_fields = p->ms->field().nrow();
#endif

    // Refuse to open if there is more than one spectral window.
    if (num_spectral_windows != 1)
    {
        fprintf(stderr, "OSKAR can read Measurement Sets with one spectral "
                "window only. Use 'split' or 'mstransform' in CASA to select "
                "the spectral window first.\n");
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }

    // Refuse to open if there is more than one field.
    if (num_fields != 1)
    {
        fprintf(stderr, "OSKAR can read Measurement Sets with one target "
                "field only. Use 'split' or 'mstransform' in CASA to select "
                "the target field first.\n");
        fflush(stderr);
        oskar_ms_close(p);
        return 0;
    }

#ifdef OSKAR_MS_NEW
    // Get the data dimensions.
    Table pol(p->ms->tableName() + "/POLARIZATION", Table::Old);
    ScalarColumn<Int> numCorr(pol, "NUM_CORR");
    if (pol.nrow() > 0)
    {
        p->num_pols = numCorr.get(0);
    }
    if (num_spectral_windows > 0)
    {
        ScalarColumn<Int> numChan(spw, "NUM_CHAN");
        ArrayColumn<Double> chanFreq(spw, "CHAN_FREQ");
        ArrayColumn<Double> chanWidth(spw, "CHAN_WIDTH");
        p->num_channels = numChan.get(0);
        p->freq_start_hz = (chanFreq.get(0))(IPosition(1, 0));
        p->freq_inc_hz = (chanWidth.get(0))(IPosition(1, 0));
    }
    Table antenna(p->ms->tableName() + "/ANTENNA", Table::Old);
    p->num_stations = antenna.nrow();
    if (p->ms->nrow() > 0)
    {
        p->time_inc_sec = p->msmc.interval.get(0);
    }

    // Get the phase centre.
    if (num_fields > 0)
    {
        ArrayColumn<Double> phaseDir(field, "PHASE_DIR");
        Matrix<Double> dir;
        phaseDir.get(0, dir);
        if (dir.nrow() > 1)
        {
            p->phase_centre_rad[0] = dir(0, 0);
            p->phase_centre_rad[1] = dir(1, 0);
        }
    }

    // Get the time range.
    Table observation(p->ms->tableName() + "/OBSERVATION", Table::Old);
    ArrayColumn<Double> timeRange(observation, "TIME_RANGE");
    if (observation.nrow() > 0)
    {
        timeRange.get(0, range);
    }
#else
    // Get the data dimensions.
    if (p->ms->polarization().nrow() > 0)
        p->num_pols = p->msc->polarization().numCorr().get(0);
    if (num_spectral_windows > 0)
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
    if (num_fields > 0)
    {
        Vector<MDirection> dir;
        p->msc->field().phaseDirMeasCol().get(0, dir, true);
        if (dir.size() > 0)
        {
            Vector<Double> v = dir(0).getAngle().getValue();
            p->phase_centre_rad[0] = v(0);
            p->phase_centre_rad[1] = v(1);
        }
    }

    // Get the time range.
    if (p->msc->observation().nrow() > 0)
        p->msc->observation().timeRange().get(0, range);
#endif
    p->start_time = range[0];
    p->end_time = range[1];

    return p;
}

oskar_MeasurementSet* oskar_ms_open(const char* filename)
{
    return oskar_ms_open_private(filename, false);
}

oskar_MeasurementSet* oskar_ms_open_readonly(const char* filename)
{
    return oskar_ms_open_private(filename, true);
}
