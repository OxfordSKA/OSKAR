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

#include "ms/MsCreate.h"

#include <ms/MeasurementSets.h>
#include <tables/Tables.h>

using namespace casa;

namespace oskar {

/**
 * @details
 * Constructs an empty measurement set with the given filename.
 *
 * @param[in] filename The Measurement Set filename to use.
 */
MsCreate::MsCreate(const char* filename,
        double mjdStart, double exposure, double interval)
{
    // Initialise members.
    _nBlocksAdded = 0;
    _mjdTimeStart = mjdStart;
    _exposure = exposure;
    _interval = interval;

    // Create the table descriptor and use it to set up a new main table.
    TableDesc desc = MS::requiredTableDesc();
    MS::addColumnToDesc(desc, MS::DATA); // For visibilities.
    MS::addColumnToDesc(desc, MS::MODEL_DATA);
    MS::addColumnToDesc(desc, MS::CORRECTED_DATA);
    // NOTE: http://listmgr.cv.nrao.edu/pipermail/casa-framework/2010-August/001861.html
    //MS::addColumnToDesc(desc, MS::IMAGING_WEIGHT);
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
}

/**
 * @details
 * Destroys the MsCreate class.
 */
MsCreate::~MsCreate()
{
    // Add rows to required subtables.
    fillObservation();

    // Select all bands and channels for imager.
    Matrix<Int> selection(2, _ms->spectralWindow().nrow());
    selection.row(0) = 0;
    selection.row(1) = _msc->spectralWindow().numChan().getColumn();
    ArrayColumn<Complex> mcd(*_ms, MS::columnName(MS::MODEL_DATA));
    mcd.rwKeywordSet().define("CHANNEL_SELECTION", selection);

    // Delete object references.
    delete _msmc;
    delete _msc;
    delete _ms;
}

/**
 * @details
 * Adds the supplied list of antenna positions to the ANTENNA table.
 */
void MsCreate::addAntennas(int na, float* /*ax*/, float* /*ay*/, float* /*az*/)
{
    // Add rows to the ANTENNA subtable.
    int startRow = _ms->antenna().nrow();
    _ms->antenna().addRow(na);
    Vector<Double> pos(3, 0.0);
    for (int a = 0; a < na; ++a) {
        int row = a + startRow;
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

/**
 * @details
 * Assumes the reference frequency is the centre of the whole band.
 * From that it calculates the centre frequency of each channel.
 */
void MsCreate::addBand(int np, int nc, double refFreq, double chanWidth)
{
    Vector<double> chanWidths(nc, chanWidth);
    Vector<double> chanFreqs(nc);
    double start = refFreq - (nc - 1) * chanWidth / 2.0;
    for (int c = 0; c < nc; ++c) {
        chanFreqs(c) = start + c * chanWidth;
    }
    addBand(np, nc, refFreq, chanFreqs, chanWidths);
}

/**
 * @details
 * Adds the given field to the FIELD table.
 */
void MsCreate::addField(double ra, double dec, const char* /*name = 0*/)
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
}

/**
 * @details
 * Adds the given list of visibilities.
 */
void MsCreate::addVisibilities(int nv, float* vu, float* vv, float* vw,
        float* vis, int* ant1, int* ant2)
{
    // Allocate storage for a (u,v,w) coordinate,
    // a visibility, a visibility weight, and a flag.
    int npol = _npol[0];
    Vector<Double> uvw(3);
    Matrix<Complex> vism(npol, 1);
    Matrix<Bool> flag(npol, 1, false);
    Vector<Float> weight(npol, 1.0);
    Matrix<Float> imagingWeight(npol, 1, 1.0);
    Vector<Float> sigma(npol, 1.0);

    // Add enough rows to the main table.
    int startRow = _ms->nrow();
    _ms->addRow(nv);

    // Loop over rows / visibilities.
    for (int v = 0; v < nv; ++v) {
        int row = v + startRow;

        // Add the u,v,w coordinates.
        uvw(0) = vu[v]; uvw(1) = vv[v]; uvw(2) = vw[v];
        _msmc->uvw().put(row, uvw);

        // Add the visibilities.
        vism = Complex(vis[2*v], vis[2*v + 1]);
        _msmc->data().put(row, vism);
        _msmc->modelData().put(row, vism);
        _msmc->correctedData().put(row, vism);

        // Add the antenna pairs.
        _msmc->antenna1().put(row, ant1[v]);
        _msmc->antenna2().put(row, ant2[v]);

        // Add remaining meta-data.
        _msmc->flag().put(row, flag);
        _msmc->weight().put(row, weight);
        _msmc->sigma().put(row, sigma);
        //NOTE: http://listmgr.cv.nrao.edu/pipermail/casa-framework/2010-August/001861.html
        //_msmc->imagingWeight().put(row, imagingWeight);
        _msmc->exposure().put(row, _exposure);
        _msmc->interval().put(row, _interval);
        _msmc->time().put(row, _mjdTimeStart * 86400
                + (_nBlocksAdded * _interval));
        _msmc->timeCentroid().put(row, _mjdTimeStart * 86400
                + (_nBlocksAdded * _interval));
    }
    _nBlocksAdded++;
}

/**
 * @details
 * Adds the given band.
 * The number of polarisations should be 1, 2 or 4,
 * and the number of channels should be > 0.
 */
void MsCreate::addBand(int np, int nc, double refFrequency,
        const Vector<double>& chanFreqs, const Vector<double>& chanWidths)
{
    // Check if this number of polarisations already exists.
    int polid = -1, nb = _npol.size();
    for (int b = 0; b < nb; ++b) {
        if (np == _npol[b]) {
            polid = _polid[b];
            break;
        }
    }
    // If not, add an entry to the POLARIZATION subtable.
    if (polid < 0) {
        addPolarisation(np);
        polid = _ms->polarization().nrow() - 1;
    }

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

    // Store the band parameters.
    _npol.push_back(np);
    _nchan.push_back(nc);
    _polid.push_back(polid);
}

/**
 * @details
 * Adds the given number of polarisations to the Measurement Set
 * by adding a row to the POLARIZATION sub-table.
 *
 * @param[in] np Number of polarisations.
 */
void MsCreate::addPolarisation(int np)
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

/**
 * @details
 * Fills the meta-data for the OBSERVATION sub-table.
 */
void MsCreate::fillObservation()
{
    Vector<String> corrSchedule(1);
    Vector<Double> timeRange(2);
    timeRange(0) = _mjdTimeStart * 86400;
    timeRange(1) = _mjdTimeStart * 86400 + (_nBlocksAdded * _interval);
    double releaseDate = timeRange(1) + 365.25 * 86400;

    // Fill observation columns.
    _ms->observation().addRow();
    _msc->observation().telescopeName().put (0, "OSKAR");
    _msc->observation().timeRange().put (0, timeRange);
    _msc->observation().schedule().put (0, corrSchedule);
    _msc->observation().project().put (0, "OSKAR");
    _msc->observation().releaseDate().put (0, releaseDate);
}

} // namespace oskar
