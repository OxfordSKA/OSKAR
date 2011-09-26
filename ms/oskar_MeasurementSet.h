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

#ifndef OSKAR_MEASUREMENT_SET_H_
#define OSKAR_MEASUREMENT_SET_H_

/**
 * @file oskar_MeasurementSet.h
 */

// CASA declarations.
namespace casa {
class MeasurementSet;
class MSColumns;
class MSMainColumns;
}

#include <casa/Arrays/Vector.h>

/**
 * @brief
 * Utility class containing functionality for manipulating a
 * Measurement Set.
 *
 * @details
 * This class contains utility functions that manipulate a Measurement Set.
 * It can be used as follows:
 * <code>
       // Create the oskar_MeasurementSet object.
       oskar_MeasurementSet ms(exposure, interval);

       // Create or open an existing measurement set.
       ms.create("filename.ms"); // or ms.open("filename.ms");

       // Add the antenna positions.
       ms.addAntennas(na, ax, ay, az);

       // Add the Right Ascension & Declination of field centre.
       ms.addField(0, 0, "name");

       // Add a polarisation (n_pol = 1, 2 or 4).
       ms.addPolarisation(n_pol);

       // Add frequency channel (polid = 0).
       ms.addBand(polid, n_chan, 400e6, 1.0);

       // Add the visibilities.
       // Note that u,v,w coordinates are in metres.
       ms.addVisibilities(n_pol, n_chan, n_row, u, v, w, vis,
               ant1, ant2, times);
 * </endcode>
 *
 */
class oskar_MeasurementSet
{
public:
    /**
     * @brief Constructs the class.
     *
     * @details
     * The constructor is used to set common parameters, such as the EXPOSURE
     * and INTERVAL values.
     *
     * @param[in] exposure The exposure length per visibility, in seconds.
     * @param[in] interval The interval length per visibility, in seconds.
     */
    oskar_MeasurementSet(double exposure, double interval);

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
     * @param[in] na The number of antennas to add.
     * @param[in] ax The antenna x positions.
     * @param[in] ay The antenna y positions.
     * @param[in] az The antenna z positions.
     */
    void addAntennas(int na, const double* ax, const double* ay,
            const double* az);

    /**
     * @brief Adds a band to the Measurement Set.
     *
     * @details
     * Assumes the reference frequency is the centre of the whole band.
     * From that it calculates the centre frequency of each channel.
     *
     * @param[in] polid The corresponding polarisation ID (assume 0).
     * @param[in] nc The number of channels in the band.
     * @param[in] refFrequency The frequency at the centre of channel 0, in Hz.
     * @param[in] chanWidth The width of each channel in Hz.
     */
    void addBand(int polid, int nc, double refFrequency, double chanWidth);

    /**
     * @brief Adds a field to the FIELD table.
     *
     * @details
     * Adds the given field to the FIELD table.
     *
     * @param[in] ra The Right Ascension of the field centre in radians.
     * @param[in] dec The Declination of the field centre in radians.
     * @param[in] name An optional string containing the name of the field.
     */
    void addField(double ra, double dec, const char* name = 0);

    /**
     * @brief Adds a polarisation to the POLARIZATION subtable.
     *
     * @details
     * Adds the given number of polarisations to the Measurement Set
     * by adding a row to the POLARIZATION sub-table.
     *
     * The number of polarisations should be 1, 2 or 4.
     *
     * @param[in] np Number of polarisations.
     */
    void addPolarisation(int np);

    /**
     * @details
     * Adds visibility data to the main table.
     *
     * @details
     * This method adds the given block of visibility data to the main table of
     * the Measurement Set. The dimensionality of the complex \p vis data block
     * is \p n_pol x \p n_chan x \p n_row, with \p n_pol the fastest varying
     * dimension, then \p n_chan, and finally \p n_row.
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
     * @param[in] n_pol  Number of polarisations.
     * @param[in] n_chan Number of channels.
     * @param[in] n_row  Number of rows to add to the main table (see note).
     * @param[in] u      Baseline u-coordinates, in metres (size n_row).
     * @param[in] v      Baseline v-coordinate, in metres (size n_row).
     * @param[in] w      Baseline w-coordinate, in metres (size n_row).
     * @param[in] vis    Matrix of complex visibilities per row (see note).
     * @param[in] ant1   Indices of antenna 1 for each baseline (size n_row).
     * @param[in] ant2   Indices of antenna 2 for each baseline (size n_row).
     * @param[in] times  Timestamp of each visibility block (size n_row).
     */
    void addVisibilities(int n_pol, int n_chan, int n_row, const double* u,
            const double* v, const double* w, const double* vis,
            const int* ant1, const int* ant2, const double* times);

    /**
     * @brief Creates a new Measurement Set.
     *
     * @details
     * Creates a new, empty Measurement Set with the given name.
     *
     * @param[in] filename The filename to use.
     */
    void create(const char* filename);

    /**
     * @brief Opens an existing Measurement Set.
     *
     * @details
     * Opens an existing Measurement Set.
     */
    void open(const char* filename);

protected:
    /**
     * @brief Cleanup routine.
     *
     * @details
     * Cleanup routine to delete objects.
     */
    void close();

    /**
     * @brief Adds a band to the Measurement Set.
     *
     * @details
     * Adds the given band.
     * The number of channels should be > 0.
     *
     * This is called by the public method of the same name.
     */
    void addBand(int polid, int nc, double refFrequency,
            const casa::Vector<double>& chanFreqs,
            const casa::Vector<double>& chanWidths);

protected:
    casa::MeasurementSet* _ms;   ///< Pointer to the Measurement Set.
    casa::MSColumns* _msc;       ///< Pointer to the sub-tables.
    casa::MSMainColumns* _msmc;  ///< Pointer to the main columns.
    double _exposure;            ///< For visibility data.
    double _interval;            ///< For visibility data.
};

#endif // OSKAR_MEASUREMENT_SET_H_
