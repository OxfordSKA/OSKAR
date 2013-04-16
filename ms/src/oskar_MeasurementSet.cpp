/*
 * Copyright (c) 2011-2013, The University of Oxford
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
#include <casa/Arrays/Vector.h>

using namespace casa;

/*=============================================================================
 * Constructor & Destructor
 *---------------------------------------------------------------------------*/

oskar_MeasurementSet::oskar_MeasurementSet() : ms_(0), msc_(0), msmc_(0)
{
}

oskar_MeasurementSet::~oskar_MeasurementSet()
{
    close();
}

/*=============================================================================
 * Public Members
 *---------------------------------------------------------------------------*/

void oskar_MeasurementSet::addAntennas(int num_antennas, const double* x,
        const double* y, const double* z, int num_receptors)
{
    if (!ms_ || !msc_) return;

    // Add rows to the ANTENNA subtable.
    int startRow = ms_->antenna().nrow();
    ms_->antenna().addRow(num_antennas);
    Vector<Double> pos(3, 0.0);
    for (int a = 0; a < num_antennas; ++a)
    {
        int row = a + startRow;
        pos(0) = x[a]; pos(1) = y[a]; pos(2) = z[a];
        msc_->antenna().position().put(row, pos);
        msc_->antenna().mount().put(row, "ALT-AZ");
        msc_->antenna().dishDiameter().put(row, 1);
        msc_->antenna().flagRow().put(row, false);
    }
    setAntennaFeeds(num_antennas, num_receptors);
}

void oskar_MeasurementSet::addAntennas(int num_antennas, const float* x,
        const float* y, const float* z, int num_receptors)
{
    if (!ms_ || !msc_) return;

    // Add rows to the ANTENNA subtable.
    int startRow = ms_->antenna().nrow();
    ms_->antenna().addRow(num_antennas);
    Vector<Double> pos(3, 0.0);
    for (int a = 0; a < num_antennas; ++a)
    {
        int row = a + startRow;
        pos(0) = x[a]; pos(1) = y[a]; pos(2) = z[a];
        msc_->antenna().position().put(row, pos);
        msc_->antenna().mount().put(row, "ALT-AZ");
        msc_->antenna().dishDiameter().put(row, 1);
        msc_->antenna().flagRow().put(row, false);
    }
    setAntennaFeeds(num_antennas, num_receptors);
}

void oskar_MeasurementSet::addBand(int polid, int num_channels, double ref_freq,
        double chan_width)
{
    Vector<double> chanWidths(num_channels, chan_width);
    Vector<double> chanFreqs(num_channels);
    //double start = refFreq - (nc - 1) * chanWidth / 2.0;
    for (int c = 0; c < num_channels; ++c)
    {
        chanFreqs(c) = ref_freq + c * chan_width;
    }
    addBand(polid, num_channels, ref_freq, chanFreqs, chanWidths);
}

void oskar_MeasurementSet::addField(double ra, double dec, const char* name)
{
    if (!ms_ || !msc_) return;

    // Set up the field info.
    MVDirection radec(Quantity(ra, "rad"), Quantity(dec, "rad"));
    Vector<MDirection> direction(1);
    direction(0) = MDirection(radec, MDirection::J2000);

    // Add a row to the FIELD table.
    int row = ms_->field().nrow();
    ms_->field().addRow();
    msc_->field().delayDirMeasCol().put(row, direction);
    msc_->field().phaseDirMeasCol().put(row, direction);
    msc_->field().referenceDirMeasCol().put(row, direction);
    if (name)
        msc_->field().name().put(row, String(name));
}

void oskar_MeasurementSet::addPolarisation(int num_pols)
{
    if (!ms_ || !msc_) return;

    // Set up the correlation type, based on number of polarisations.
    Vector<Int> corrType(num_pols);
    corrType(0) = Stokes::XX;
    if (num_pols == 2)
    {
        corrType(1) = Stokes::YY;
    }
    else if (num_pols == 4)
    {
        corrType(1) = Stokes::XY;
        corrType(2) = Stokes::YX;
        corrType(3) = Stokes::YY;
    }

    // Set up the correlation product, based on number of polarisations.
    Matrix<Int> corrProduct(2, num_pols);
    for (int i = 0; i < num_pols; ++i)
    {
        corrProduct(0, i) = Stokes::receptor1(Stokes::type(corrType(i)));
        corrProduct(1, i) = Stokes::receptor2(Stokes::type(corrType(i)));
    }

    // Create a new row, and fill the columns.
    int row = ms_->polarization().nrow();
    ms_->polarization().addRow();
    msc_->polarization().corrType().put(row, corrType);
    msc_->polarization().corrProduct().put(row, corrProduct);
    msc_->polarization().numCorr().put(row, num_pols);
}

void oskar_MeasurementSet::addVisibilities(int num_pols, int num_channels,
        int num_rows, const double* u, const double* v, const double* w,
        const double* vis, const int* ant1, const int* ant2,
        double exposure, double interval, const double* times)
{
    if (!ms_ || !msc_ || !msmc_) return;

    // Allocate storage for a (u,v,w) coordinate,
    // a visibility matrix, a visibility weight, and a flag matrix.
    Vector<Double> uvw(3);
    Matrix<Complex> vis_data(num_pols, num_channels);
    Matrix<Bool> flag(num_pols, num_channels, false);
    Vector<Float> weight(num_pols, 1.0);
    Vector<Float> sigma(num_pols, 1.0);

    // Add enough rows to the main table.
    int start_row = ms_->nrow();
    ms_->addRow(num_rows);

    // Loop over rows / visibilities.
    for (int r = 0; r < num_rows; ++r)
    {
        int row = r + start_row;

        // Add the u,v,w coordinates.
        uvw(0) = u[r]; uvw(1) = v[r]; uvw(2) = w[r];
        msmc_->uvw().put(row, uvw);

        // Get a pointer to the start of the visibility matrix for this row.
        const double* vis_row = vis + (2 * num_pols * num_channels) * r;

        // Fill the visibility matrix (polarisation and channel data).
        for (int c = 0; c < num_channels; ++c)
        {
            for (int p = 0; p < num_pols; ++p)
            {
                int b = 2 * (p + c * num_pols);
                vis_data(p, c) = Complex(vis_row[b], vis_row[b + 1]);
            }
        }

        // Add the visibilities.
        msmc_->data().put(row, vis_data);

        // Add the antenna pairs.
        msmc_->antenna1().put(row, ant1[r]);
        msmc_->antenna2().put(row, ant2[r]);

        // Add remaining meta-data.
        msmc_->flag().put(row, flag);
        msmc_->weight().put(row, weight);
        msmc_->sigma().put(row, sigma);
        msmc_->exposure().put(row, exposure);
        msmc_->interval().put(row, interval);
        msmc_->time().put(row, times[r]);
        msmc_->timeCentroid().put(row, times[r]);
    }
    setTimeRange(times[0], times[num_rows - 1]);
}

void oskar_MeasurementSet::addVisibilities(int num_pols, int num_channels,
        int num_rows, const float* u, const float* v, const float* w,
        const float* vis, const int* ant1, const int* ant2,
        double exposure, double interval, const float* times)
{
    if (!ms_ || !msc_ || !msmc_) return;

    // Allocate storage for a (u,v,w) coordinate,
    // a visibility matrix, a visibility weight, and a flag matrix.
    Vector<Double> uvw(3);
    Matrix<Complex> vis_data(num_pols, num_channels);
    Matrix<Bool> flag(num_pols, num_channels, false);
    Vector<Float> weight(num_pols, 1.0);
    Vector<Float> sigma(num_pols, 1.0);

    // Add enough rows to the main table.
    int start_row = ms_->nrow();
    ms_->addRow(num_rows);

    // Loop over rows / visibilities.
    for (int r = 0; r < num_rows; ++r)
    {
        int row = r + start_row;

        // Add the u,v,w coordinates.
        uvw(0) = u[r]; uvw(1) = v[r]; uvw(2) = w[r];
        msmc_->uvw().put(row, uvw);

        // Get a pointer to the start of the visibility matrix for this row.
        const float* vis_row = vis + (2 * num_pols * num_channels) * r;

        // Fill the visibility matrix (polarisation and channel data).
        for (int c = 0; c < num_channels; ++c)
        {
            for (int p = 0; p < num_pols; ++p)
            {
                int b = 2 * (p + c * num_pols);
                vis_data(p, c) = Complex(vis_row[b], vis_row[b + 1]);
            }
        }

        // Add the visibilities.
        msmc_->data().put(row, vis_data);

        // Add the antenna pairs.
        msmc_->antenna1().put(row, ant1[r]);
        msmc_->antenna2().put(row, ant2[r]);

        // Add remaining meta-data.
        msmc_->flag().put(row, flag);
        msmc_->weight().put(row, weight);
        msmc_->sigma().put(row, sigma);
        msmc_->exposure().put(row, exposure);
        msmc_->interval().put(row, interval);
        msmc_->time().put(row, times[r]);
        msmc_->timeCentroid().put(row, times[r]);
    }
    setTimeRange(times[0], times[num_rows - 1]);
}

void oskar_MeasurementSet::close()
{
    // Delete object references.
    if (msmc_)
    {
        delete msmc_;
        msmc_ = 0;
    }
    if (msc_)
    {
        delete msc_;
        msc_ = 0;
    }
    if (ms_)
    {
        delete ms_;
        ms_ = 0;
    }
}

void oskar_MeasurementSet::create(const char* filename)
{
    // Close any existing Measurment Set.
    if (ms_ || msc_ || msmc_)
        close();

    // Create the table descriptor and use it to set up a new main table.
    TableDesc desc = MS::requiredTableDesc();
    MS::addColumnToDesc(desc, MS::DATA); // For visibilities.
    SetupNewTable newTab(filename, desc, Table::New);

    // Create the MeasurementSet.
    ms_ = new MeasurementSet(newTab);

    // Create SOURCE sub-table.
    TableDesc descSource = MSSource::requiredTableDesc();
    MSSource::addColumnToDesc(descSource, MSSource::REST_FREQUENCY);
    MSSource::addColumnToDesc(descSource, MSSource::POSITION);
    SetupNewTable sourceSetup(ms_->sourceTableName(), descSource, Table::New);
    ms_->rwKeywordSet().defineTable(MS::keywordName(MS::SOURCE),
                   Table(sourceSetup));

    // Create all required default subtables.
    ms_->createDefaultSubtables(Table::New);

    // Create the MSMainColumns and MSColumns objects for accessing data
    // in the main table and subtables.
    msc_ = new MSColumns(*ms_);
    msmc_ = new MSMainColumns(*ms_);

    // Add a row to the OBSERVATION subtable.
    ms_->observation().addRow();
    Vector<String> corrSchedule(1);
    Vector<Double> timeRange(2, 0.0);
    msc_->observation().schedule().put(0, corrSchedule);
    msc_->observation().project().put(0, "OSKAR");
    msc_->observation().telescopeName().put(0, "OSKAR " OSKAR_VERSION_STR);
    msc_->observation().timeRange().put(0, timeRange);
    setTimeRange(0.0, 1.0);
}

void oskar_MeasurementSet::open(const char* filename)
{
    // Close any existing Measurment Set.
    if (ms_ || msc_ || msmc_)
        close();

    // Create the MeasurementSet.
    ms_ = new MeasurementSet(filename, Table::Update);

    // Create the MSMainColumns and MSColumns objects for accessing data
    // in the main table and subtables.
    msc_ = new MSColumns(*ms_);
    msmc_ = new MSMainColumns(*ms_);
}


/*=============================================================================
 * Protected Members
 *---------------------------------------------------------------------------*/

void oskar_MeasurementSet::addBand(int polid, int num_channels,
        double refFrequency, const Vector<double>& chanFreqs,
        const Vector<double>& chanWidths)
{
    if (!ms_ || !msc_) return;

    // Add a row to the DATA_DESCRIPTION subtable.
    int row = ms_->dataDescription().nrow();
    ms_->dataDescription().addRow();
    msc_->dataDescription().spectralWindowId().put(row, row);
    msc_->dataDescription().polarizationId().put(row, polid);
    msc_->dataDescription().flagRow().put(row, false);
    ms_->dataDescription().flush();

    // Get total bandwidth from maximum and minimum.
    Vector<double> startFreqs = chanFreqs - chanWidths / 2.0;
    Vector<double> endFreqs = chanFreqs + chanWidths / 2.0;
    double totalBandwidth = max(endFreqs) - min(startFreqs);

    // Add a row to the SPECTRAL_WINDOW sub-table.
    ms_->spectralWindow().addRow();
    msc_->spectralWindow().measFreqRef().put(row, MFrequency::TOPO);
    msc_->spectralWindow().chanFreq().put(row, chanFreqs);
    msc_->spectralWindow().refFrequency().put(row, refFrequency);
    msc_->spectralWindow().chanWidth().put(row, chanWidths);
    msc_->spectralWindow().effectiveBW().put(row, chanWidths);
    msc_->spectralWindow().resolution().put(row, chanWidths);
    msc_->spectralWindow().flagRow().put(row, false);
    msc_->spectralWindow().freqGroup().put(row, 0);
    msc_->spectralWindow().freqGroupName().put(row, "");
    msc_->spectralWindow().ifConvChain().put(row, 0);
    msc_->spectralWindow().name().put(row, "");
    msc_->spectralWindow().netSideband().put(row, 0);
    msc_->spectralWindow().numChan().put(row, num_channels);
    msc_->spectralWindow().totalBandwidth().put(row, totalBandwidth);
    ms_->spectralWindow().flush();
}

void oskar_MeasurementSet::setAntennaFeeds(int num_antennas, int num_receptors)
{
    if (!ms_ || !msc_) return;

    // Determine constants for the FEED subtable.
    Matrix<Double> feedOffset(2, num_receptors, 0.0);
    Matrix<Complex> feedResponse(num_receptors, num_receptors, Complex(0.0, 0.0));
    Vector<String> feedType(num_receptors);
    feedType(0) = "X";
    if (num_receptors > 1) feedType(1) = "Y";
    Vector<Double> feedAngle(num_receptors, 0.0);

    // Fill the FEED subtable (required).
    int startRow = ms_->feed().nrow();
    ms_->feed().addRow(num_antennas);
    for (int a = 0; a < num_antennas; ++a)
    {
        int row = a + startRow;
        msc_->feed().antennaId().put(row, a);
        msc_->feed().beamOffset().put(row, feedOffset);
        msc_->feed().polarizationType().put(row, feedType);
        msc_->feed().polResponse().put(row, feedResponse);
        msc_->feed().receptorAngle().put(row, feedAngle);
        msc_->feed().numReceptors().put(row, num_receptors);
    }
}

void oskar_MeasurementSet::setTimeRange(double start_time, double end_time)
{
    // Get the old time range.
    Vector<Double> oldTimeRange(2, 0.0);
    msc_->observation().timeRange().get(0, oldTimeRange);

    // Compute the new time range.
    Vector<Double> timeRange(2, 0.0);
    timeRange[0] = (oldTimeRange[0] <= 0.0) ? start_time : oldTimeRange[0];
    timeRange[1] = (end_time > oldTimeRange[1]) ? end_time : oldTimeRange[1];
    double releaseDate = timeRange[1] + 365.25 * 86400.0;

    // Fill observation columns.
    msc_->observation().timeRange().put(0, timeRange);
    msc_->observation().releaseDate().put(0, releaseDate);
}
