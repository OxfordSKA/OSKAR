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

#include <cstdlib>
#include <cstdio>

using namespace casa;

oskar_MeasurementSet* oskar_ms_open(const char* filename)
{
    oskar_MeasurementSet* p = (oskar_MeasurementSet*)
            calloc(1, sizeof(oskar_MeasurementSet));

    try
    {
        // Create the MeasurementSet. Storage managers are recreated as needed.
        p->ms = new MeasurementSet(filename,
                TableLock(TableLock::PermanentLocking), Table::Update);

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
    p->phase_centre_ra = 0.0;
    p->phase_centre_dec = 0.0;
    if (p->ms->field().nrow() > 0)
    {
        Vector<MDirection> dir;
        p->msc->field().phaseDirMeasCol().get(0, dir, true);
        if (dir.size() > 0)
        {
            Vector<Double> v = dir(0).getAngle().getValue();
            p->phase_centre_ra = v(0);
            p->phase_centre_dec = v(1);
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
