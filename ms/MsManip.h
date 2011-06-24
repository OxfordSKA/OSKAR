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

#ifndef OSKAR_MS_MANIP_MS_H_
#define OSKAR_MS_MANIP_MS_H_

/**
 * @file MsManip.h
 */

// CASA declarations.
namespace casa {
class MeasurementSet;
class MSColumns;
class MSMainColumns;
}

#include <casa/Arrays/Vector.h>

namespace oskar {

/**
 * @brief
 * Utility class containing functionality for manipulating a
 * Measurement Set.
 *
 * @details
 * This class contains utility functions that manipulate a Measurement Set.
 * It can be used as follows:
 * <code>
       // Create the MsManip object.
       MsManip ms(start, exposure, interval);

       // Create or open an existing measurement set.
       ms.create("filename.ms"); // or ms.open("filename.ms");

       // Add the antenna positions.
       ms.addAntennas(na, ax, ay, az);

       // Add the Right Ascension & Declination of field centre.
       ms.addField(0, 0);

       // Add a polarisation (np = 1).
       ms.addPolarisation(np);

       // Add frequency band (polid = 0).
       ms.addBand(polid, 1, 400e6, 1.0);

       // Add the visibilities.
       // Note that u,v,w coordinates are in metres.
       ms.addVisibilities(np, nv, u, v, w, vis, ant1, ant2);
 * </endcode>
 *
 */
class MsManip
{
public:
    /// Constructs an empty measurement set with the given filename.
    MsManip(double mjdStart, double exposure, double interval);

    /// Destroys the MsManip class.
    ~MsManip();

    /// Adds antennas to the ANTENNA table.
    void addAntennas(int na, const double* ax, const double* ay,
            const double* az);

    /// Adds a band to the Measurement Set.
    void addBand(int polid, int nc, double refFrequency, double chanWidth);

    /// Adds a field to the FIELD table.
    void addField(double ra, double dec, const char* name = 0);

    /// Adds a polarisation to the POLARIZATION subtable.
    void addPolarisation(int np);

    /// Adds visibilities to the main table.
    void addVisibilities(int np, int nv, const double* vu, const double* vv,
            const double* vw, const double* vis, const int* ant1,
            const int* ant2);

    /// Adds visibilities to the main table.
    void addVisibilities(int np, int nv, const double* vu, const double* vv,
            const double* vw, const double* vis, const int* ant1,
            const int* ant2, const double* times);

    /// Creates a new measurement set.
    void create(const char* filename);

    /// Opens an existing measurement set.
    void open(const char* filename);

protected:
    /// Cleanup routine.
    void close();

    /// Adds a band to the Measurement Set.
    void addBand(int polid, int nc, double refFrequency,
            const casa::Vector<double>& chanFreqs,
            const casa::Vector<double>& chanWidths);

    /// Updates the time range in the OBSERVATION sub-table.
    void updateTimeRange();

protected:
    casa::MeasurementSet* _ms;   ///< Pointer to the created Measurement Set.
    casa::MSColumns* _msc;       ///< Pointer to the sub-tables.
    casa::MSMainColumns* _msmc;  ///< Pointer to the main columns.
    int _nBlocksAdded;           ///< The number of visibility blocks added.
    double _mjdTimeStart;        ///< Modified Julian Date of the start time.
    double _exposure;            ///< For visibility data.
    double _interval;            ///< For visibility data.
};

} // namespace oskar

#endif // OSKAR_MS_MANIP_MS_H_
