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

#ifndef OSKAR_MEASUREMENT_SET_H_
#define OSKAR_MEASUREMENT_SET_H_

/**
 * @file oskar_MeasurementSet.h
 */

#include <oskar_global.h>
#include <stddef.h>

// CASA declarations.
namespace casa {
class MeasurementSet;
class MSColumns;
class MSMainColumns;
template <class T> class Vector;
}

/**
 * @brief
 * Utility class containing functionality for manipulating a
 * Measurement Set.
 *
 * @details
 * This class contains utility functions that manipulate a Measurement Set.
 * It can be used as follows:
   <code>
       // Create the oskar_MeasurementSet object.
       oskar_MeasurementSet ms;

       // Create or open an existing measurement set.
       ms.create("filename.ms"); // or ms.open("filename.ms");

       // Add the antenna positions.
       ms.addAntennas(num_antennas, x, y, z);

       // Add the Right Ascension & Declination of field centre.
       ms.addField(0, 0, "name");

       // Add a polarisation (num_pols = 1, 2 or 4).
       ms.addPolarisation(num_pols);

       // Add frequency channel (polid = 0).
       ms.addBand(polid, num_channels, 400e6, 1.0);

       // Add the visibilities.
       // Note that u,v,w coordinates are in metres.
       ms.addVisibilities(num_pols, num_channels, num_rows, u, v, w, vis,
               ant1, ant2, exposure, interval, times);
   </endcode>
 *
 */
class OSKAR_MS_EXPORT oskar_MeasurementSet
{
public:
    /**
     * @brief Constructs the class.
     *
     * @details
     * Constructs the class.
     */
    oskar_MeasurementSet();

    /**
     * @brief Destroys the class.
     *
     * @details
     * Destroys the class and frees any resources used by it.
     */
    ~oskar_MeasurementSet();

    /**
     * @brief Adds antennas to the ANTENNA table.
     *
     * @details
     * Adds the supplied list of antenna positions to the ANTENNA table.
     *
     * @param[in] num_antennas  The number of antennas to add.
     * @param[in] x             The antenna x positions.
     * @param[in] y             The antenna y positions.
     * @param[in] z             The antenna z positions.
     * @param[in] num_receptors The number of receptors in the feeds.
     */
    void addAntennas(int num_antennas, const double* x, const double* y,
            const double* z, int num_receptors = 2);

    /**
     * @brief Adds antennas to the ANTENNA table.
     *
     * @details
     * Adds the supplied list of antenna positions to the ANTENNA table.
     *
     * @param[in] num_antennas  The number of antennas to add.
     * @param[in] x             The antenna x positions.
     * @param[in] y             The antenna y positions.
     * @param[in] z             The antenna z positions.
     * @param[in] num_receptors The number of receptors in the feeds.
     */
    void addAntennas(int num_antennas, const float* x, const float* y,
            const float* z, int num_receptors = 2);

    /**
     * @brief Adds a band to the Measurement Set.
     *
     * @details
     * Assumes the reference frequency is the centre of the whole band.
     * From that it calculates the centre frequency of each channel.
     *
     * @param[in] polid         The corresponding polarisation ID (assume 0).
     * @param[in] num_channels  The number of channels in the band.
     * @param[in] ref_freq      The frequency at the centre of channel 0, in Hz.
     * @param[in] chan_width    The width of each channel in Hz.
     */
    void addBand(int polid, int num_channels, double ref_freq,
            double chan_width);

    /**
     * @brief Adds a field to the FIELD table.
     *
     * @details
     * Adds the given field to the FIELD table.
     *
     * @param[in] ra   The Right Ascension of the field centre in radians.
     * @param[in] dec  The Declination of the field centre in radians.
     * @param[in] name An optional string containing the name of the field.
     */
    void addField(double ra, double dec, const char* name = 0);

    /**
     * @brief Adds the run log to the HISTORY table.
     *
     * @details
     * Adds the supplied run log string to the HISTORY table.
     * The log string is split into lines, and each is added as its own
     * HISTORY entry. The origin of each will be "LOG".
     *
     * @param[in] str   The log file, as a single string.
     * @param[in] size  The length of the string.
     */
    void addLog(const char* str, size_t size);

    /**
     * @brief Adds the contents of the settings file to the HISTORY table.
     *
     * @details
     * Adds the supplied settings file string to the HISTORY table.
     * The settings string is split into lines, and these are entered to
     * the APP_PARAMS column as a single HISTORY entry.
     * The history message will read "OSKAR settings file", and the origin
     * will be "SETTINGS".
     *
     * @param[in] str   The settings file, as a single string.
     * @param[in] size  The length of the string.
     */
    void addSettings(const char* str, size_t size);

    /**
     * @brief Adds a polarisation to the POLARIZATION subtable.
     *
     * @details
     * Adds the given number of polarisations to the Measurement Set
     * by adding a row to the POLARIZATION sub-table.
     *
     * The number of polarisations should be 1, 2 or 4.
     *
     * @param[in] num_pols Number of polarisations.
     */
    void addPolarisation(int num_pols);

    /**
     * @details
     * Adds visibility data to the main table.
     *
     * @details
     * This method adds the given block of visibility data to the main table of
     * the Measurement Set. The dimensionality of the complex \p vis data block
     * is \p num_pols x \p num_channels x \p num_rows, with \p num_pols the
     * fastest varying dimension, then \p num_channels, and finally \p num_rows.
     *
     * Each row of the main table holds data from a single baseline for a
     * single time stamp, so the number of rows is given by the number of
     * baselines multiplied by the number of times. The complex visibilities
     * are therefore understood to be given per polarisation, per channel and
     * per baseline (and repeated as many times as required).
     *
     * The times are given in units of (MJD) * 86400, i.e. seconds since
     * Julian date 2400000.5.
     *
     * Thus (for C-ordered memory), the layout of \p vis corresponding to two
     * time snapshots, for a three-element interferometer with four
     * polarisations and two channels would be:
     *
     * time0,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     *
     * @param[in] num_pols     Number of polarisations.
     * @param[in] num_channels Number of channels.
     * @param[in] num_rows     Number of rows to add to the main table (see note).
     * @param[in] u            Baseline u-coordinates, in metres (size num_rows).
     * @param[in] v            Baseline v-coordinate, in metres (size num_rows).
     * @param[in] w            Baseline w-coordinate, in metres (size num_rows).
     * @param[in] vis          Matrix of complex visibilities per row (see note).
     * @param[in] ant1         Indices of antenna 1 for each baseline (size num_rows).
     * @param[in] ant2         Indices of antenna 2 for each baseline (size num_rows).
     * @param[in] exposure     The exposure length per visibility, in seconds.
     * @param[in] interval     The interval length per visibility, in seconds.
     * @param[in] times        Timestamp of each visibility block (size num_rows).
     */
    void addVisibilities(int num_pols, int num_channels, int num_rows,
            const double* u, const double* v, const double* w,
            const double* vis, const int* ant1, const int* ant2,
            double exposure, double interval, const double* times);

    /**
     * @details
     * Replaces visibility data in the main table.
     *
     * @details
     * This method puts the given block of visibility data in the main table of
     * the Measurement Set. The dimensionality of the complex \p vis data block
     * is \p num_pols x \p num_channels x \p num_rows, with \p num_pols the
     * fastest varying dimension, then \p num_channels, and finally \p num_rows.
     *
     * Each row of the main table holds data from a single baseline for a
     * single time stamp, so the number of rows is given by the number of
     * baselines multiplied by the number of times. The complex visibilities
     * are therefore understood to be given per polarisation, per channel and
     * per baseline (and repeated as many times as required).
     *
     * The times are given in units of (MJD) * 86400, i.e. seconds since
     * Julian date 2400000.5.
     *
     * Thus (for C-ordered memory), the layout of \p vis corresponding to two
     * time snapshots, for a three-element interferometer with four
     * polarisations and two channels would be:
     *
     * time0,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     *
     * @param[in] start_row    The start row index of the Measurement Set.
     * @param[in] num_pols     Number of polarisations.
     * @param[in] num_channels Number of channels.
     * @param[in] num_rows     Number of rows to add to the main table (see note).
     * @param[in] u            Baseline u-coordinates, in metres (size num_rows).
     * @param[in] v            Baseline v-coordinate, in metres (size num_rows).
     * @param[in] w            Baseline w-coordinate, in metres (size num_rows).
     * @param[in] vis          Matrix of complex visibilities per row (see note).
     * @param[in] ant1         Indices of antenna 1 for each baseline (size num_rows).
     * @param[in] ant2         Indices of antenna 2 for each baseline (size num_rows).
     * @param[in] exposure     The exposure length per visibility, in seconds.
     * @param[in] interval     The interval length per visibility, in seconds.
     * @param[in] times        Timestamp of each visibility block (size num_rows).
     */
    void putVisibilities(int start_row, int num_pols, int num_channels,
            int num_rows, const double* u, const double* v, const double* w,
            const double* vis, const int* ant1, const int* ant2,
            double exposure, double interval, const double* times);

    /**
     * @details
     * Adds visibility data to the main table.
     *
     * @details
     * This method adds the given block of visibility data to the main table of
     * the Measurement Set. The dimensionality of the complex \p vis data block
     * is \p num_pols x \p num_channels x \p num_rows, with \p num_pols the
     * fastest varying dimension, then \p num_channels, and finally \p num_rows.
     *
     * Each row of the main table holds data from a single baseline for a
     * single time stamp, so the number of rows is given by the number of
     * baselines multiplied by the number of times. The complex visibilities
     * are therefore understood to be given per polarisation, per channel and
     * per baseline (and repeated as many times as required).
     *
     * The times are given in units of (MJD) * 86400, i.e. seconds since
     * Julian date 2400000.5.
     *
     * Thus (for C-ordered memory), the layout of \p vis corresponding to two
     * time snapshots, for a three-element interferometer with four
     * polarisations and two channels would be:
     *
     * time0,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     *
     * @param[in] num_pols     Number of polarisations.
     * @param[in] num_channels Number of channels.
     * @param[in] num_rows     Number of rows to add to the main table (see note).
     * @param[in] u            Baseline u-coordinates, in metres (size num_rows).
     * @param[in] v            Baseline v-coordinate, in metres (size num_rows).
     * @param[in] w            Baseline w-coordinate, in metres (size num_rows).
     * @param[in] vis          Matrix of complex visibilities per row (see note).
     * @param[in] ant1         Indices of antenna 1 for each baseline (size num_rows).
     * @param[in] ant2         Indices of antenna 2 for each baseline (size num_rows).
     * @param[in] exposure     The exposure length per visibility, in seconds.
     * @param[in] interval     The interval length per visibility, in seconds.
     * @param[in] times        Timestamp of each visibility block (size num_rows).
     */
    void addVisibilities(int num_pols, int num_channels, int num_rows,
            const float* u, const float* v, const float* w,
            const float* vis, const int* ant1, const int* ant2,
            double exposure, double interval, const float* times);

    /**
     * @details
     * Replaces visibility data to the main table.
     *
     * @details
     * This method puts the given block of visibility data in the main table of
     * the Measurement Set. The dimensionality of the complex \p vis data block
     * is \p num_pols x \p num_channels x \p num_rows, with \p num_pols the
     * fastest varying dimension, then \p num_channels, and finally \p num_rows.
     *
     * Each row of the main table holds data from a single baseline for a
     * single time stamp, so the number of rows is given by the number of
     * baselines multiplied by the number of times. The complex visibilities
     * are therefore understood to be given per polarisation, per channel and
     * per baseline (and repeated as many times as required).
     *
     * The times are given in units of (MJD) * 86400, i.e. seconds since
     * Julian date 2400000.5.
     *
     * Thus (for C-ordered memory), the layout of \p vis corresponding to two
     * time snapshots, for a three-element interferometer with four
     * polarisations and two channels would be:
     *
     * time0,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time0,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-1
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant0-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     * time1,ant1-2
     *   pol0,ch0  pol1,ch0  pol2,ch0  pol3,ch0
     *   pol0,ch1  pol1,ch1  pol2,ch1  pol3,ch1
     *
     * @param[in] start_row    The start row index of the Measurement Set.
     * @param[in] num_pols     Number of polarisations.
     * @param[in] num_channels Number of channels.
     * @param[in] num_rows     Number of rows to add to the main table (see note).
     * @param[in] u            Baseline u-coordinates, in metres (size num_rows).
     * @param[in] v            Baseline v-coordinate, in metres (size num_rows).
     * @param[in] w            Baseline w-coordinate, in metres (size num_rows).
     * @param[in] vis          Matrix of complex visibilities per row (see note).
     * @param[in] ant1         Indices of antenna 1 for each baseline (size num_rows).
     * @param[in] ant2         Indices of antenna 2 for each baseline (size num_rows).
     * @param[in] exposure     The exposure length per visibility, in seconds.
     * @param[in] interval     The interval length per visibility, in seconds.
     * @param[in] times        Timestamp of each visibility block (size num_rows).
     */
    void putVisibilities(int start_row, int num_pols, int num_channels,
            int num_rows, const float* u, const float* v, const float* w,
            const float* vis, const int* ant1, const int* ant2,
            double exposure, double interval, const float* times);

    /**
     * @brief Cleanup routine.
     *
     * @details
     * Cleanup routine to delete objects.
     */
    void close();

    /**
     * @brief Creates a new Measurement Set.
     *
     * @details
     * Creates a new, empty Measurement Set with the given name.
     *
     * @param[in] filename The filename to use.
     */
    void create(const char* filename, int num_pols, int num_channels,
            int num_stations);

    /**
     * @brief Opens an existing Measurement Set.
     *
     * @details
     * Opens an existing Measurement Set.
     */
    void open(const char* filename);

    /**
     * @brief Sets the number of rows in the main table.
     *
     * @details
     * Sets the number of rows in the main table to be at least as large
     * as the value given.
     */
    void setNumRows(int num);

    /**
     * @brief Sets the time range of the data.
     *
     * @details
     * Sets the time range of the data in the main table. This is typically
     * used after multiple calls to putVisibilities().
     */
    void setTimeRange(double start_time, double last_time);

protected:
    /**
     * @brief Adds a band to the Measurement Set.
     *
     * @details
     * Adds the given band.
     * The number of channels should be > 0.
     *
     * This is called by the public method of the same name.
     */
    void addBand(int polid, int num_channels, double refFrequency,
            const casa::Vector<double>& chanFreqs,
            const casa::Vector<double>& chanWidths);

    void setAntennaFeeds(int num_antennas, int num_receptors);

protected:
    casa::MeasurementSet* ms_;   ///< Pointer to the Measurement Set.
    casa::MSColumns* msc_;       ///< Pointer to the sub-tables.
    casa::MSMainColumns* msmc_;  ///< Pointer to the main columns.
};

#endif // OSKAR_MEASUREMENT_SET_H_
