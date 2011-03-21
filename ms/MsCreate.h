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

#include "ms/MsManip.h"

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

       // Add a polarisation.
       ms.addPolarisation(np);

       // Add frequency band.
       ms.addBand(polid, 1, 400e6, 1.0);

       // Add the visibilities.
       // Note that u,v,w coordinates are in metres.
       ms.addVisibilities(np, nv, u, v, w, vis, ant1, ant2);
 * </endcode>
 */
class MsCreate : public MsManip
{
public:
    /// Constructs an empty measurement set with the given filename.
    MsCreate(const char* filename, double mjdStart, double exposure,
            double interval);

    /// Destroys the MsCreate class.
    ~MsCreate();
};

} // namespace oskar

#endif // OSKAR_MS_CREATE_MS_H_
