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

#ifndef OSKAR_MS_CREATE_MS_H_
#define OSKAR_MS_CREATE_MS_H_

/**
 * @file MsCreate.h
 */

// CASA declarations.
namespace casa {
class MeasurementSet;
class MSColumns;
class MSMainColumns;
}

namespace oskar {

/**
 * @brief
 * Utility class for creating a Measurement Set.
 *
 * @details
 * This class is used to easily create a Measurement Set.
 * It should be used as follows:
 * <code>
       // Create the MsCreate object, passing it the filename.
       MsCreate ms("simple.ms", start, exposure, interval);

       // Add the antenna positions.
       ms.addAntennas(na, ax, ay, az);

       // Add the Right Ascension & Declination of field centre.
       ms.addField(0, 0);

       // Add frequency band.
       ms.addBand(1, 1, 400e6, 1.0);

       // Add the visibilities.
       // Note that u,v,w coordinates are in metres.
       ms.addVisibilities(nv, u, v, w, vis);
 * </endcode>
 */
class MsCreate
{
public:
    /// Constructs an empty measurement set with the given filename.
    MsCreate(const char* filename, double mjdStart, double exposure, double interval);

    /// Destroys the MsCreate class.
    ~MsCreate();

    /// Adds antennas to the ANTENNA table.
    void addAntennas(int na, float* ax, float* ay, float* az);

    /// Adds a band to the Measurement Set.
    void addBand(int np, int nc, double refFrequency, double chanWidth);

    /// Adds a field to the FIELD table.
    void addField(double ra, double dec, const char* name = 0);

    /// Adds visibilities to the main table.
    void addVisibilities(int nv, float* u, float* v, float* w, float* vis,
        int* ant1, int* ant2);

private:
    /// Adds a band to the Measurement Set.
    void addBand(int np, int nc, double refFrequency,
            const Vector<double>& chanFreqs,
            const Vector<double>& chanWidths);

    /// Adds the given number of polarisations.
    void addPolarisation(int np);

    /// Fills the observation sub-table.
    void fillObservation();

private:
    casa::MeasurementSet* _ms;   ///< Pointer to the created Measurement Set.
    casa::MSColumns* _msc;       ///< Pointer to the sub-tables.
    casa::MSMainColumns* _msmc;  ///< Pointer to the main columns.
    std::vector<int> _npol;      ///< Number of polarisations in each band.
    std::vector<int> _nchan;     ///< Number of channels in each band.
    std::vector<int> _polid;     ///< Polarisation ID for each band.
    int _nBlocksAdded;           ///< The number of visibility blocks added.
    double _mjdTimeStart;        ///< Modified Julian Date of the start time.
    double _exposure;            ///< For visibility data.
    double _interval;            ///< For visibility data.
};

} // namespace oskar

#endif // OSKAR_MS_CREATE_MS_H_
