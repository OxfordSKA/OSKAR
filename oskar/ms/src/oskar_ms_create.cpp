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
#include <ctime>

using namespace casa;

static void oskar_ms_add_band(oskar_MeasurementSet* p, int pol_id,
        unsigned int num_channels, double ref_freq,
        const Vector<double>& chan_freqs,
        const Vector<double>& chan_widths);
static void oskar_ms_add_pol(oskar_MeasurementSet* p, unsigned int num_pols);

oskar_MeasurementSet* oskar_ms_create(const char* file_name,
        const char* app_name, unsigned int num_stations,
        unsigned int num_channels, unsigned int num_pols, double freq_start_hz,
        double freq_inc_hz, int write_autocorr, int write_crosscorr)
{
    oskar_MeasurementSet* p = (oskar_MeasurementSet*)
            calloc(1, sizeof(oskar_MeasurementSet));

    // Create the table descriptor and use it to set up a new main table.
    TableDesc desc = MS::requiredTableDesc();
    MS::addColumnToDesc(desc, MS::DATA, 2); // Visibilities (2D: pol, chan).
    desc.rwColumnDesc(MS::columnName(MS::DATA)).
            rwKeywordSet().define("UNIT", "Jy");
    IPosition dataShape(2, num_pols, num_channels);
    IPosition weightShape(1, num_pols);
    desc.rwColumnDesc(MS::columnName(MS::DATA)).setShape(dataShape);
    desc.rwColumnDesc(MS::columnName(MS::FLAG)).setShape(dataShape);
    desc.rwColumnDesc(MS::columnName(MS::WEIGHT)).setShape(weightShape);
    desc.rwColumnDesc(MS::columnName(MS::SIGMA)).setShape(weightShape);
    Vector<String> tsmNames(1);
    tsmNames[0] = MS::columnName(MS::DATA);
    desc.defineHypercolumn("TiledData", 3, tsmNames);
    tsmNames[0] = MS::columnName(MS::FLAG);
    desc.defineHypercolumn("TiledFlag", 3, tsmNames);
    tsmNames[0] = MS::columnName(MS::UVW);
    desc.defineHypercolumn("TiledUVW", 2, tsmNames);
    tsmNames[0] = MS::columnName(MS::WEIGHT);
    desc.defineHypercolumn("TiledWeight", 2, tsmNames);
    tsmNames[0] = MS::columnName(MS::SIGMA);
    desc.defineHypercolumn("TiledSigma", 2, tsmNames);
    try
    {
        unsigned int num_baselines = 0;

        if (write_autocorr && write_crosscorr)
            num_baselines = num_stations * (num_stations + 1) / 2;
        else if (!write_autocorr && write_crosscorr)
            num_baselines = num_stations * (num_stations - 1) / 2;
        else if (write_autocorr && !write_crosscorr)
            num_baselines = num_stations;
        else
        {
            oskar_ms_close(p);
            return 0;
        }

        SetupNewTable tab(file_name, desc, Table::New);

        // Create the default storage managers.
        IncrementalStMan incrStorageManager("ISMData");
        tab.bindAll(incrStorageManager);
        StandardStMan stdStorageManager("SSMData", 32768, 32768);
        tab.bindColumn(MS::columnName(MS::ANTENNA1), stdStorageManager);
        tab.bindColumn(MS::columnName(MS::ANTENNA2), stdStorageManager);

        // Create tiled column storage manager for UVW column.
        IPosition uvwTileShape(2, 3, 2 * num_baselines);
        TiledColumnStMan uvwStorageManager("TiledUVW", uvwTileShape);
        tab.bindColumn(MS::columnName(MS::UVW), uvwStorageManager);

        // Create tiled column storage managers for WEIGHT and SIGMA columns.
        IPosition weightTileShape(2, num_pols, 2 * num_baselines);
        TiledColumnStMan weightStorageManager("TiledWeight", weightTileShape);
        tab.bindColumn(MS::columnName(MS::WEIGHT), weightStorageManager);
        IPosition sigmaTileShape(2, num_pols, 2 * num_baselines);
        TiledColumnStMan sigmaStorageManager("TiledSigma", sigmaTileShape);
        tab.bindColumn(MS::columnName(MS::SIGMA), sigmaStorageManager);

        // Create tiled column storage managers for DATA and FLAG columns.
        IPosition dataTileShape(3, num_pols, num_channels, 2 * num_baselines);
        TiledColumnStMan dataStorageManager("TiledData", dataTileShape);
        tab.bindColumn(MS::columnName(MS::DATA), dataStorageManager);
        IPosition flagTileShape(3, num_pols, num_channels, 16 * num_baselines);
        TiledColumnStMan flagStorageManager("TiledFlag", flagTileShape);
        tab.bindColumn(MS::columnName(MS::FLAG), flagStorageManager);

        // Create the Measurement Set.
        p->ms = new MeasurementSet(tab, TableLock(TableLock::PermanentLocking));

        // Create SOURCE sub-table.
        TableDesc descSource = MSSource::requiredTableDesc();
        MSSource::addColumnToDesc(descSource, MSSource::REST_FREQUENCY);
        MSSource::addColumnToDesc(descSource, MSSource::POSITION);
        SetupNewTable sourceSetup(p->ms->sourceTableName(),
                descSource, Table::New);
        p->ms->rwKeywordSet().defineTable(MS::keywordName(MS::SOURCE),
                Table(sourceSetup));

        // Create all required default subtables.
        p->ms->createDefaultSubtables(Table::New);

        // Create the MSMainColumns and MSColumns objects for accessing data
        // in the main table and subtables.
        p->msc = new MSColumns(*(p->ms));
        p->msmc = new MSMainColumns(*(p->ms));
        p->app_name = (char*) realloc(p->app_name, strlen(app_name) + 1);
        strcpy(p->app_name, app_name);
    }
    catch (...)
    {
        fprintf(stderr, "Error creating Measurement Set!\n");
        oskar_ms_close(p);
        return 0;
    }

    // Add a row to the OBSERVATION subtable.
    const char* username = getenv("USERNAME");
    if (!username)
        username = getenv("USER");
    p->start_time = DBL_MAX;
    p->end_time = -DBL_MAX;
    p->ms->observation().addRow();
    Vector<String> corrSchedule(1);
    Vector<Double> timeRange(2, 0.0);
    p->msc->observation().schedule().put(0, corrSchedule);
    p->msc->observation().project().put(0, "");
    p->msc->observation().observer().put(0, username);
    p->msc->observation().telescopeName().put(0, app_name);
    p->msc->observation().timeRange().put(0, timeRange);
    oskar_ms_set_time_range(p);

    // Add polarisation ID.
    oskar_ms_add_pol(p, num_pols);

    // Add a dummy field to size the FIELD table.
    oskar_ms_set_phase_centre(p, 0, 0.0, 0.0);

    // Set up the band.
    Vector<double> chan_widths(num_channels, freq_inc_hz);
    Vector<double> chan_freqs(num_channels);
    //double start = ref_freq - (num_channels - 1) * chan_width / 2.0;
    for (unsigned int c = 0; c < num_channels; ++c)
    {
        chan_freqs(c) = freq_start_hz + c * freq_inc_hz;
    }
    oskar_ms_add_band(p, 0, num_channels, freq_start_hz, chan_freqs, chan_widths);

    // Get a string containing the current system time.
    char time_str[80];
    time_t unix_time;
    unix_time = std::time(NULL);
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)",
            std::localtime(&unix_time));

    // Add a row to the HISTORY subtable.
    String msg = String("Measurement Set created at ") + String(time_str);
    oskar_ms_add_history(p, app_name, msg.c_str(), msg.size());

    // Set the private data.
    p->num_pols = num_pols;
    p->num_channels = num_channels;
    p->num_stations = num_stations;
    p->num_receptors = 2; // By default.
    p->freq_start_hz = freq_start_hz;
    p->freq_inc_hz = freq_inc_hz;

    // Fill the ANTENNA table.
    p->ms->antenna().addRow(num_stations);
    Vector<Double> pos(3, 0.0);
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        p->msc->antenna().position().put(a, pos);
        p->msc->antenna().mount().put(a, "FIXED");
        p->msc->antenna().dishDiameter().put(a, 1);
        p->msc->antenna().flagRow().put(a, false);
    }

    // Fill the FEED table.
    Matrix<Double> feedOffset(2, p->num_receptors, 0.0);
    Matrix<Complex> feedResponse(p->num_receptors, p->num_receptors,
            Complex(0.0, 0.0));
    Vector<String> feedType(p->num_receptors);
    feedType(0) = "X";
    if (p->num_receptors > 1) feedType(1) = "Y";
    Vector<Double> feedAngle(p->num_receptors, 0.0);
    p->ms->feed().addRow(num_stations);
    for (unsigned int a = 0; a < num_stations; ++a)
    {
        p->msc->feed().antennaId().put(a, a);
        p->msc->feed().beamOffset().put(a, feedOffset);
        p->msc->feed().polarizationType().put(a, feedType);
        p->msc->feed().polResponse().put(a, feedResponse);
        p->msc->feed().receptorAngle().put(a, feedAngle);
        p->msc->feed().numReceptors().put(a, p->num_receptors);
    }

    return p;
}

void oskar_ms_add_band(oskar_MeasurementSet* p, int pol_id,
        unsigned int num_channels, double ref_freq,
        const Vector<double>& chan_freqs,
        const Vector<double>& chan_widths)
{
    if (!p->ms || !p->msc) return;

    // Add a row to the DATA_DESCRIPTION subtable.
    unsigned int row = p->ms->dataDescription().nrow();
    p->ms->dataDescription().addRow();
    p->msc->dataDescription().spectralWindowId().put(row, row);
    p->msc->dataDescription().polarizationId().put(row, pol_id);
    p->msc->dataDescription().flagRow().put(row, false);

    // Get total bandwidth from maximum and minimum.
    Vector<double> startFreqs = chan_freqs - chan_widths / 2.0;
    Vector<double> endFreqs = chan_freqs + chan_widths / 2.0;
    double totalBandwidth = max(endFreqs) - min(startFreqs);

    // Add a row to the SPECTRAL_WINDOW sub-table.
    p->ms->spectralWindow().addRow();
    MSSpWindowColumns& s = p->msc->spectralWindow();
    s.measFreqRef().put(row, MFrequency::TOPO);
    s.chanFreq().put(row, chan_freqs);
    s.refFrequency().put(row, ref_freq);
    s.chanWidth().put(row, chan_widths);
    s.effectiveBW().put(row, chan_widths);
    s.resolution().put(row, chan_widths);
    s.flagRow().put(row, false);
    s.freqGroup().put(row, 0);
    s.freqGroupName().put(row, "");
    s.ifConvChain().put(row, 0);
    s.name().put(row, "");
    s.netSideband().put(row, 0);
    s.numChan().put(row, num_channels);
    s.totalBandwidth().put(row, totalBandwidth);
}

void oskar_ms_add_pol(oskar_MeasurementSet* p, unsigned int num_pols)
{
    if (!p->ms || !p->msc) return;

    // Set up the correlation type, based on number of polarisations.
    Vector<Int> corr_type(num_pols);
    corr_type(0) = Stokes::XX; // Can't be Stokes I if num_pols = 1! (Throws exception.)
    if (num_pols == 2)
    {
        corr_type(1) = Stokes::YY;
    }
    else if (num_pols == 4)
    {
        corr_type(1) = Stokes::XY;
        corr_type(2) = Stokes::YX;
        corr_type(3) = Stokes::YY;
    }

    // Set up the correlation product, based on number of polarisations.
    Matrix<Int> corr_product(2, num_pols);
    for (unsigned int i = 0; i < num_pols; ++i)
    {
        corr_product(0, i) = Stokes::receptor1(Stokes::type(corr_type(i)));
        corr_product(1, i) = Stokes::receptor2(Stokes::type(corr_type(i)));
    }

    // Create a new row, and fill the columns.
    unsigned int row = p->ms->polarization().nrow();
    p->ms->polarization().addRow();
    p->msc->polarization().corrType().put(row, corr_type);
    p->msc->polarization().corrProduct().put(row, corr_product);
    p->msc->polarization().numCorr().put(row, num_pols);
}
