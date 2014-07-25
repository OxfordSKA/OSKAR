/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <string>
#include <sstream>
#include <vector>
#include <ctime>

using namespace casa;

/*=============================================================================
 * Local (static) functions
 *---------------------------------------------------------------------------*/

static std::vector<std::string> split(const std::string& s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> v;
    while (std::getline(ss, item, delim))
    {
        v.push_back(item);
    }
    return v;
}

double current_utc_to_mjd()
{
    int a, y, m, jdn;
    double day_fraction;
    time_t unix_time;
    struct tm* time_s;

    // Get system UTC.
    unix_time = std::time(NULL);
    time_s = std::gmtime(&unix_time);

    // Compute Julian Day Number (Note: all integer division).
    // Note that tm_mon is in range 0-11, so must add 1.
    a = (14 - (time_s->tm_mon + 1)) / 12;
    y = (time_s->tm_year + 1900) + 4800 - a;
    m = (time_s->tm_mon + 1) + 12 * a - 3;
    jdn = time_s->tm_mday + (153 * m + 2) / 5 + (365 * y) + (y / 4) - (y / 100)
            + (y / 400) - 32045;

    // Compute day fraction.
    day_fraction = time_s->tm_hour / 24.0 + time_s->tm_min / 1440.0 +
            time_s->tm_sec / 86400.0;
    return jdn + day_fraction - 2400000.5 - 0.5;
}

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

void oskar_MeasurementSet::addLog(const char* str, size_t size)
{
    if (!ms_ || !msc_ || !str) return;

    // Construct a string from the char array and split on each newline.
    std::vector<std::string> v = split(std::string(str, size), '\n');

    // Add to the HISTORY table.
    int num_lines = v.size();
    int row = ms_->history().nrow();
    ms_->history().addRow(num_lines);
    double current_utc = 86400.0 * current_utc_to_mjd();
    for (int i = 0; i < num_lines; ++i)
    {
        msc_->history().message().put(row + i, String(v[i]));
        msc_->history().application().put(row + i, "OSKAR " OSKAR_VERSION_STR);
        msc_->history().priority().put(row + i, "INFO");
        msc_->history().origin().put(row + i, "LOG");
        msc_->history().time().put(row + i, current_utc);
        msc_->history().observationId().put(row + i, -1);
        msc_->history().appParams().put(row + i, Vector<String>()); // Required!
        msc_->history().cliCommand().put(row + i, Vector<String>()); // Required!
    }
}

void oskar_MeasurementSet::addSettings(const char* str, size_t size)
{
    if (!ms_ || !msc_ || !str) return;

    // Construct a string from the char array and split on each newline.
    std::vector<std::string> v = split(std::string(str, size), '\n');

    // Fill a CASA vector with the settings file contents.
    int num_lines = v.size();
    Vector<String> vec(num_lines);
    for (int i = 0; i < num_lines; ++i)
    {
        vec(i) = v[i];
    }

    // Add to the HISTORY table.
    int row = ms_->history().nrow();
    ms_->history().addRow();
    msc_->history().appParams().put(row, vec);
    msc_->history().message().put(row, "OSKAR settings file");
    msc_->history().application().put(row, "OSKAR " OSKAR_VERSION_STR);
    msc_->history().priority().put(row, "INFO");
    msc_->history().origin().put(row, "SETTINGS");
    msc_->history().time().put(row, 86400.0 * current_utc_to_mjd());
    msc_->history().observationId().put(row, -1);
    msc_->history().cliCommand().put(row, Vector<String>()); // Required!
}

void oskar_MeasurementSet::addPolarisation(int num_pols)
{
    if (!ms_ || !msc_) return;

    // Set up the correlation type, based on number of polarisations.
    Vector<Int> corrType(num_pols);
    corrType(0) = Stokes::XX; // Can't be Stokes I if num_pols = 1! (Throws exception.)
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

    // Add enough rows to the main table.
    int start_row = ms_->nrow();
    ms_->addRow(num_rows);

    putVisibilities(start_row, num_pols, num_channels, num_rows,
            u, v, w, vis, ant1, ant2, exposure, interval, times);
    setTimeRange(times[0], times[num_rows - 1]);
}

void oskar_MeasurementSet::putVisibilities(int start_row, int num_pols,
        int num_channels, int num_rows, const double* u, const double* v,
        const double* w, const double* vis, const int* ant1, const int* ant2,
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

    // Get references to columns.
    ArrayColumn<Double>& col_uvw = msmc_->uvw();
    ArrayColumn<Complex>& col_data = msmc_->data();
    ScalarColumn<Int>& col_antenna1 = msmc_->antenna1();
    ScalarColumn<Int>& col_antenna2 = msmc_->antenna2();
    ArrayColumn<Bool>& col_flag = msmc_->flag();
    ArrayColumn<Float>& col_weight = msmc_->weight();
    ArrayColumn<Float>& col_sigma = msmc_->sigma();
    ScalarColumn<Double>& col_exposure = msmc_->exposure();
    ScalarColumn<Double>& col_interval = msmc_->interval();
    ScalarColumn<Double>& col_time = msmc_->time();
    ScalarColumn<Double>& col_timeCentroid = msmc_->timeCentroid();

    // Loop over rows / visibilities.
    for (int r = 0; r < num_rows; ++r)
    {
        int row = r + start_row;

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

        // Write the data to the Measurement Set.
        uvw(0) = u[r]; uvw(1) = v[r]; uvw(2) = w[r];
        col_uvw.put(row, uvw);
        col_antenna1.put(row, ant1[r]);
        col_antenna2.put(row, ant2[r]);
        col_data.put(row, vis_data);
        col_flag.put(row, flag);
        col_weight.put(row, weight);
        col_sigma.put(row, sigma);
        col_exposure.put(row, exposure);
        col_interval.put(row, interval);
        col_time.put(row, times[r]);
        col_timeCentroid.put(row, times[r]);
    }
}

void oskar_MeasurementSet::addVisibilities(int num_pols, int num_channels,
        int num_rows, const float* u, const float* v, const float* w,
        const float* vis, const int* ant1, const int* ant2,
        double exposure, double interval, const float* times)
{
    if (!ms_ || !msc_ || !msmc_) return;

    // Add enough rows to the main table.
    int start_row = ms_->nrow();
    ms_->addRow(num_rows);

    putVisibilities(start_row, num_pols, num_channels, num_rows,
            u, v, w, vis, ant1, ant2, exposure, interval, times);
    setTimeRange(times[0], times[num_rows - 1]);
}

void oskar_MeasurementSet::putVisibilities(int start_row, int num_pols,
        int num_channels, int num_rows, const float* u, const float* v,
        const float* w, const float* vis, const int* ant1, const int* ant2,
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

    // Get references to columns.
    ArrayColumn<Double>& col_uvw = msmc_->uvw();
    ArrayColumn<Complex>& col_data = msmc_->data();
    ScalarColumn<Int>& col_antenna1 = msmc_->antenna1();
    ScalarColumn<Int>& col_antenna2 = msmc_->antenna2();
    ArrayColumn<Bool>& col_flag = msmc_->flag();
    ArrayColumn<Float>& col_weight = msmc_->weight();
    ArrayColumn<Float>& col_sigma = msmc_->sigma();
    ScalarColumn<Double>& col_exposure = msmc_->exposure();
    ScalarColumn<Double>& col_interval = msmc_->interval();
    ScalarColumn<Double>& col_time = msmc_->time();
    ScalarColumn<Double>& col_timeCentroid = msmc_->timeCentroid();

    // Loop over rows / visibilities.
    for (int r = 0; r < num_rows; ++r)
    {
        int row = r + start_row;

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

        // Write the data to the Measurement Set.
        uvw(0) = u[r]; uvw(1) = v[r]; uvw(2) = w[r];
        col_uvw.put(row, uvw);
        col_antenna1.put(row, ant1[r]);
        col_antenna2.put(row, ant2[r]);
        col_data.put(row, vis_data);
        col_flag.put(row, flag);
        col_weight.put(row, weight);
        col_sigma.put(row, sigma);
        col_exposure.put(row, exposure);
        col_interval.put(row, interval);
        col_time.put(row, times[r]);
        col_timeCentroid.put(row, times[r]);
    }
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

void oskar_MeasurementSet::create(const char* filename,
        int num_pols, int num_channels, int num_stations)
{
    // Close any existing Measurement Set.
    if (ms_ || msc_ || msmc_)
        close();

    // Create the table descriptor and use it to set up a new main table.
    TableDesc desc = MS::requiredTableDesc();
    MS::addColumnToDesc(desc, MS::DATA, 2); // Visibilities (2D: pol, chan).
    desc.rwColumnDesc(MS::columnName(MS::DATA)).
            rwKeywordSet().define("UNIT", "Jy");
    IPosition dataShape(2, num_pols, num_channels);
    desc.rwColumnDesc(MS::columnName(MS::DATA)).setShape(dataShape);
    desc.rwColumnDesc(MS::columnName(MS::FLAG)).setShape(dataShape);
    Vector<String> tsmNames(1);
    tsmNames[0] = MS::columnName(MS::DATA);
    desc.defineHypercolumn("TiledData", 3, tsmNames);
    tsmNames[0] = MS::columnName(MS::FLAG);
    desc.defineHypercolumn("TiledFlag", 3, tsmNames);
    tsmNames[0] = MS::columnName(MS::UVW);
    desc.defineHypercolumn("TiledUVW", 2, tsmNames);
    SetupNewTable newTab(filename, desc, Table::New);

    // Create the default storage managers.
    IncrementalStMan incrStorageManager("ISMData");
    newTab.bindAll(incrStorageManager);
    StandardStMan stdStorageManager("SSMData", 32768, 32768);
    newTab.bindColumn(MS::columnName(MS::ANTENNA1), stdStorageManager);
    newTab.bindColumn(MS::columnName(MS::ANTENNA2), stdStorageManager);

    // Create tiled column storage manager for UVW column.
    IPosition uvwTileShape(2, 3, 2 * num_stations * (num_stations - 1) / 2);
    TiledColumnStMan uvwStorageManager("TiledUVW", uvwTileShape);
    newTab.bindColumn(MS::columnName(MS::UVW), uvwStorageManager);

    // Create tiled column storage managers for DATA and FLAG columns.
    IPosition dataTileShape(3, num_pols, num_channels,
            2 * num_stations * (num_stations - 1) / 2);
    TiledColumnStMan dataStorageManager("TiledData", dataTileShape);
    newTab.bindColumn(MS::columnName(MS::DATA), dataStorageManager);
    IPosition flagTileShape(3, num_pols, num_channels,
            16 * num_stations * (num_stations - 1) / 2);
    TiledColumnStMan flagStorageManager("TiledFlag", flagTileShape);
    newTab.bindColumn(MS::columnName(MS::FLAG), flagStorageManager);

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
    msc_->observation().project().put(0, "");
    msc_->observation().telescopeName().put(0, "OSKAR " OSKAR_VERSION_STR);
    msc_->observation().timeRange().put(0, timeRange);
    setTimeRange(0.0, 1.0);

    // Get a string containing the current system time.
    char time_str[80];
    time_t unix_time;
    unix_time = std::time(NULL);
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)",
            std::localtime(&unix_time));

    // Add a row to the HISTORY subtable.
    ms_->history().addRow();
    msc_->history().message().put(0, String("Measurement Set created at ") +
            String(time_str));
    msc_->history().application().put(0, "OSKAR " OSKAR_VERSION_STR);
    msc_->history().origin().put(0, "OSKAR " OSKAR_VERSION_STR);
    msc_->history().priority().put(0, "INFO");
    msc_->history().time().put(0, 86400.0 * current_utc_to_mjd());
    msc_->history().observationId().put(0, -1);
    msc_->history().appParams().put(0, Vector<String>()); // Required!
    msc_->history().cliCommand().put(0, Vector<String>()); // Required!
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

void oskar_MeasurementSet::setNumRows(int num)
{
    if (!ms_) return;

    int old_num_rows = ms_->nrow();
    int rows_to_add = num - old_num_rows;
    if (rows_to_add <= 0) return;
    ms_->addRow(rows_to_add);
}

void oskar_MeasurementSet::setTimeRange(double start_time, double end_time)
{
    if (!msc_) return;

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
