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

#include "ms/oskar_MeasurementSet.h"

#include <ms/MeasurementSets.h>
#include <tables/Tables.h>

using namespace casa;

/*=============================================================================
 * Constructor & Destructor
 *---------------------------------------------------------------------------*/

oskar_MeasurementSet::oskar_MeasurementSet()
{
    // Initialise members.
    _ms = NULL;
    _msc = NULL;
    _msmc = NULL;
}

oskar_MeasurementSet::~oskar_MeasurementSet()
{
    // Select all bands and channels for imager (not sure if this is required?).
    Matrix<Int> selection(2, _ms->spectralWindow().nrow());
    selection.row(0) = 0;
    selection.row(1) = _msc->spectralWindow().numChan().getColumn();
    ArrayColumn<Complex> mcd(*_ms, MS::columnName(MS::MODEL_DATA));
    mcd.rwKeywordSet().define("CHANNEL_SELECTION", selection);
    close();
}

/*=============================================================================
 * Public Members
 *---------------------------------------------------------------------------*/

void oskar_MeasurementSet::addAntennas(int na, const double* ax,
        const double* ay, const double* az)
{
    // Add rows to the ANTENNA subtable.
    int startRow = _ms->antenna().nrow();
    _ms->antenna().addRow(na);
    Vector<Double> pos(3, 0.0);
    for (int a = 0; a < na; ++a) {
        int row = a + startRow;
        pos(0) = ax[a]; pos(1) = ay[a]; pos(2) = az[a];
        _msc->antenna().position().put(row, pos);
        _msc->antenna().mount().put(row, "ALT-AZ");
        _msc->antenna().dishDiameter().put(row, 1);
        _msc->antenna().flagRow().put(row, false);
    }

    // Determine constants for the FEED subtable.
    int nRec = 2;
    Matrix<Double> feedOffset(2, nRec, 0.0);
    Matrix<Complex> feedResponse(nRec, nRec, Complex(0.0, 0.0));
    Vector<String> feedType(nRec);
    feedType(0) = "X";
    if (nRec > 1) feedType(1) = "Y";
    Vector<Double> feedAngle(nRec, 0.0);

    // Fill the FEED subtable (required).
    _ms->feed().addRow(na);
    for (int a = 0; a < na; ++a) {
        _msc->feed().antennaId().put(a, a);
        _msc->feed().beamOffset().put(a, feedOffset);
        _msc->feed().polarizationType().put(a, feedType);
        _msc->feed().polResponse().put(a, feedResponse);
        _msc->feed().receptorAngle().put(a, feedAngle);
        _msc->feed().numReceptors().put(a, 1);
    }
}

void oskar_MeasurementSet::addAntennas(int na, const float* ax,
        const float* ay, const float* az)
{
    // Add rows to the ANTENNA subtable.
    int startRow = _ms->antenna().nrow();
    _ms->antenna().addRow(na);
    Vector<Double> pos(3, 0.0);
    for (int a = 0; a < na; ++a) {
        int row = a + startRow;
        pos(0) = ax[a]; pos(1) = ay[a]; pos(2) = az[a];
        _msc->antenna().position().put(row, pos);
        _msc->antenna().mount().put(row, "ALT-AZ");
        _msc->antenna().dishDiameter().put(row, 1);
        _msc->antenna().flagRow().put(row, false);
    }

    // Determine constants for the FEED subtable.
    int nRec = 2;
    Matrix<Double> feedOffset(2, nRec, 0.0);
    Matrix<Complex> feedResponse(nRec, nRec, Complex(0.0, 0.0));
    Vector<String> feedType(nRec);
    feedType(0) = "X";
    if (nRec > 1) feedType(1) = "Y";
    Vector<Double> feedAngle(nRec, 0.0);

    // Fill the FEED subtable (required).
    _ms->feed().addRow(na);
    for (int a = 0; a < na; ++a) {
        _msc->feed().antennaId().put(a, a);
        _msc->feed().beamOffset().put(a, feedOffset);
        _msc->feed().polarizationType().put(a, feedType);
        _msc->feed().polResponse().put(a, feedResponse);
        _msc->feed().receptorAngle().put(a, feedAngle);
        _msc->feed().numReceptors().put(a, 1);
    }
}

void oskar_MeasurementSet::addBand(int polid, int nc, double refFreq,
        double chanWidth)
{
    Vector<double> chanWidths(nc, chanWidth);
    Vector<double> chanFreqs(nc);
    //double start = refFreq - (nc - 1) * chanWidth / 2.0;
    for (int c = 0; c < nc; ++c) {
        chanFreqs(c) = refFreq + c * chanWidth;
    }
    addBand(polid, nc, refFreq, chanFreqs, chanWidths);
}

void oskar_MeasurementSet::addField(double ra, double dec, const char* name)
{
    // Set up the field info.
    MVDirection radec(Quantity(ra, "rad"), Quantity(dec, "rad"));
    Vector<MDirection> direction(1);
    direction(0) = MDirection(radec, MDirection::J2000);

    // Add a row to the FIELD table.
    int row = _ms->field().nrow();
    _ms->field().addRow();
    _msc->field().delayDirMeasCol().put(row, direction);
    _msc->field().phaseDirMeasCol().put(row, direction);
    _msc->field().referenceDirMeasCol().put(row, direction);
    if (name) {
        _msc->field().name().put(row, String(name));
    }
}

void oskar_MeasurementSet::addPolarisation(int np)
{
    // Set up the correlation type, based on number of polarisations.
    Vector<Int> corrType(np);
    corrType(0) = Stokes::XX;
    if (np == 2) {
        corrType(1) = Stokes::YY;
    } else if (np == 4) {
        corrType(1) = Stokes::XY;
        corrType(2) = Stokes::YX;
        corrType(3) = Stokes::YY;
    }

    // Set up the correlation product, based on number of polarisations.
    Matrix<Int> corrProduct(2, np);
    for (int i = 0; i < np; ++i) {
        corrProduct(0, i) = Stokes::receptor1(Stokes::type(corrType(i)));
        corrProduct(1, i) = Stokes::receptor2(Stokes::type(corrType(i)));
    }

    // Create a new row, and fill the columns.
    int row = _ms->polarization().nrow();
    _ms->polarization().addRow();
    _msc->polarization().corrType().put(row, corrType);
    _msc->polarization().corrProduct().put(row, corrProduct);
    _msc->polarization().numCorr().put(row, np);
}

void oskar_MeasurementSet::addVisibilities(int n_pol, int n_chan, int n_row,
        const double* u, const double* v, const double* w, const double* vis,
        const int* ant1, const int* ant2, double exposure, double interval,
        const double* times)
{
    // Allocate storage for a (u,v,w) coordinate,
    // a visibility matrix, a visibility weight, and a flag matrix.
    Vector<Double> uvw(3);
    Matrix<Complex> vis_data(n_pol, n_chan);
    Matrix<Bool> flag(n_pol, n_chan, false);
    Vector<Float> weight(n_pol, 1.0);
    Vector<Float> sigma(n_pol, 1.0);

    // Add enough rows to the main table.
    int start_row = _ms->nrow();
    _ms->addRow(n_row);

    // Loop over rows / visibilities.
    for (int r = 0; r < n_row; ++r)
    {
        int row = r + start_row;

        // Add the u,v,w coordinates.
        uvw(0) = u[r]; uvw(1) = v[r]; uvw(2) = w[r];
        _msmc->uvw().put(row, uvw);

        // Get a pointer to the start of the visibility matrix for this row.
        const double* vis_row = vis + (2 * n_pol * n_chan) * r;

        // Fill the visibility matrix (polarisation and channel data).
        for (int c = 0; c < n_chan; ++c)
        {
            for (int p = 0; p < n_pol; ++p)
            {
                int b = 2 * (p + c * n_pol);
                vis_data(p, c) = Complex(vis_row[b], vis_row[b + 1]);
            }
        }

        // Add the visibilities.
        _msmc->data().put(row, vis_data);
        _msmc->modelData().put(row, vis_data);
        _msmc->correctedData().put(row, vis_data);

        // Add the antenna pairs.
        _msmc->antenna1().put(row, ant1[r]);
        _msmc->antenna2().put(row, ant2[r]);

        // Add remaining meta-data.
        _msmc->flag().put(row, flag);
        _msmc->weight().put(row, weight);
        _msmc->sigma().put(row, sigma);
        _msmc->exposure().put(row, exposure);
        _msmc->interval().put(row, interval);
        _msmc->time().put(row, times[r]);
        _msmc->timeCentroid().put(row, times[r]);
    }

    // Get the old time range.
    Vector<Double> oldTimeRange(2);
    _msc->observation().timeRange().get(0, oldTimeRange);

    // Compute the new time range.
    Vector<Double> timeRange(2);
    timeRange[0] = (oldTimeRange[0] <= 0.0) ? times[0] : oldTimeRange[0];
    timeRange[1] = (times[n_row - 1] > oldTimeRange[1]) ? times[n_row - 1] :
            oldTimeRange[1];
    double releaseDate = timeRange[1] + 365.25 * 86400.0;

    // Fill observation columns.
    _msc->observation().timeRange().put(0, timeRange);
    _msc->observation().releaseDate().put(0, releaseDate);
}

void oskar_MeasurementSet::addVisibilities(int n_pol, int n_chan, int n_row,
        const float* u, const float* v, const float* w, const float* vis,
        const int* ant1, const int* ant2, double exposure, double interval,
        const float* times)
{
    // Allocate storage for a (u,v,w) coordinate,
    // a visibility matrix, a visibility weight, and a flag matrix.
    Vector<Double> uvw(3);
    Matrix<Complex> vis_data(n_pol, n_chan);
    Matrix<Bool> flag(n_pol, n_chan, false);
    Vector<Float> weight(n_pol, 1.0);
    Vector<Float> sigma(n_pol, 1.0);

    // Add enough rows to the main table.
    int start_row = _ms->nrow();
    _ms->addRow(n_row);

    // Loop over rows / visibilities.
    for (int r = 0; r < n_row; ++r)
    {
        int row = r + start_row;

        // Add the u,v,w coordinates.
        uvw(0) = u[r]; uvw(1) = v[r]; uvw(2) = w[r];
        _msmc->uvw().put(row, uvw);

        // Get a pointer to the start of the visibility matrix for this row.
        const float* vis_row = vis + (2 * n_pol * n_chan) * r;

        // Fill the visibility matrix (polarisation and channel data).
        for (int c = 0; c < n_chan; ++c)
        {
            for (int p = 0; p < n_pol; ++p)
            {
                int b = 2 * (p + c * n_pol);
                vis_data(p, c) = Complex(vis_row[b], vis_row[b + 1]);
            }
        }

        // Add the visibilities.
        _msmc->data().put(row, vis_data);
        _msmc->modelData().put(row, vis_data);
        _msmc->correctedData().put(row, vis_data);

        // Add the antenna pairs.
        _msmc->antenna1().put(row, ant1[r]);
        _msmc->antenna2().put(row, ant2[r]);

        // Add remaining meta-data.
        _msmc->flag().put(row, flag);
        _msmc->weight().put(row, weight);
        _msmc->sigma().put(row, sigma);
        _msmc->exposure().put(row, exposure);
        _msmc->interval().put(row, interval);
        _msmc->time().put(row, times[r]);
        _msmc->timeCentroid().put(row, times[r]);
    }

    // Get the old time range.
    Vector<Double> oldTimeRange(2);
    _msc->observation().timeRange().get(0, oldTimeRange);

    // Compute the new time range.
    Vector<Double> timeRange(2);
    timeRange[0] = (oldTimeRange[0] <= 0.0) ? times[0] : oldTimeRange[0];
    timeRange[1] = (times[n_row - 1] > oldTimeRange[1]) ? times[n_row - 1] :
            oldTimeRange[1];
    double releaseDate = timeRange[1] + 365.25 * 86400.0;

    // Fill observation columns.
    _msc->observation().timeRange().put(0, timeRange);
    _msc->observation().releaseDate().put(0, releaseDate);
}


void oskar_MeasurementSet::close()
{
    // Delete object references.
    delete _msmc;
    delete _msc;
    delete _ms;
}

void oskar_MeasurementSet::create(const char* filename)
{
    // Create the table descriptor and use it to set up a new main table.
    TableDesc desc = MS::requiredTableDesc();
    MS::addColumnToDesc(desc, MS::DATA); // For visibilities.
    MS::addColumnToDesc(desc, MS::MODEL_DATA);
    MS::addColumnToDesc(desc, MS::CORRECTED_DATA);
    SetupNewTable newTab(filename, desc, Table::New);

    // Create the MeasurementSet.
    _ms = new MeasurementSet(newTab);

    // Create SOURCE sub-table.
    TableDesc descSource = MSSource::requiredTableDesc();
    MSSource::addColumnToDesc(descSource, MSSource::REST_FREQUENCY);
    MSSource::addColumnToDesc(descSource, MSSource::POSITION);
    SetupNewTable sourceSetup(_ms->sourceTableName(), descSource, Table::New);
    _ms->rwKeywordSet().defineTable(MS::keywordName(MS::SOURCE),
                   Table(sourceSetup));

    // Create all required default subtables.
    _ms->createDefaultSubtables(Table::New);

    // Create the MSMainColumns and MSColumns objects for accessing data
    // in the main table and subtables.
    _msc = new MSColumns(*_ms);
    _msmc = new MSMainColumns(*_ms);

    // Add a row to the OBSERVATION subtable.
    _ms->observation().addRow();
    Vector<String> corrSchedule(1);
    Vector<Double> timeRange(2);
    timeRange(0) = 0;
    timeRange(1) = 1;
    double releaseDate = timeRange(1) + 365.25 * 86400;
    _msc->observation().telescopeName().put (0, "OSKAR");
    _msc->observation().timeRange().put (0, timeRange);
    _msc->observation().schedule().put (0, corrSchedule);
    _msc->observation().project().put (0, "OSKAR");
    _msc->observation().releaseDate().put (0, releaseDate);
}

void oskar_MeasurementSet::open(const char* filename)
{
    // Create the MeasurementSet.
    _ms = new MeasurementSet(filename, Table::Update);

    // Create the MSMainColumns and MSColumns objects for accessing data
    // in the main table and subtables.
    _msc = new MSColumns(*_ms);
    _msmc = new MSMainColumns(*_ms);
}


/*=============================================================================
 * Protected Members
 *---------------------------------------------------------------------------*/

void oskar_MeasurementSet::addBand(int polid, int nc, double refFrequency,
        const Vector<double>& chanFreqs, const Vector<double>& chanWidths)
{
    // Add a row to the DATA_DESCRIPTION subtable.
    int row = _ms->dataDescription().nrow();
    _ms->dataDescription().addRow();
    _msc->dataDescription().spectralWindowId().put(row, row);
    _msc->dataDescription().polarizationId().put(row, polid);
    _msc->dataDescription().flagRow().put(row, false);
    _ms->dataDescription().flush();

    // Get total bandwidth from maximum and minimum.
    Vector<double> startFreqs = chanFreqs - chanWidths / 2.0;
    Vector<double> endFreqs = chanFreqs + chanWidths / 2.0;
    double totalBandwidth = max(endFreqs) - min(startFreqs);

    // Add a row to the SPECTRAL_WINDOW sub-table.
    _ms->spectralWindow().addRow();
    _msc->spectralWindow().measFreqRef().put(row, MFrequency::TOPO);
    _msc->spectralWindow().chanFreq().put(row, chanFreqs);
    _msc->spectralWindow().refFrequency().put(row, refFrequency);
    _msc->spectralWindow().chanWidth().put(row, chanWidths);
    _msc->spectralWindow().effectiveBW().put(row, chanWidths);
    _msc->spectralWindow().resolution().put(row, chanWidths);
    _msc->spectralWindow().flagRow().put(row, false);
    _msc->spectralWindow().freqGroup().put(row, 0);
    _msc->spectralWindow().freqGroupName().put(row, "");
    _msc->spectralWindow().ifConvChain().put(row, 0);
    _msc->spectralWindow().name().put(row, "");
    _msc->spectralWindow().netSideband().put(row, 0);
    _msc->spectralWindow().numChan().put(row, nc);
    _msc->spectralWindow().totalBandwidth().put(row, totalBandwidth);
    _ms->spectralWindow().flush();
}
