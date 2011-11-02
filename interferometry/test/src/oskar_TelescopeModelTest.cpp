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

#include "interferometry/test/oskar_TelescopeModelTest.h"
#include "interferometry/oskar_TelescopeModel.h"

#include <cmath>
#include <cstdio>

/**
 * @details
 * Tests filling a telescope model, and copying it to the GPU and back.
 */
void oskar_TelescopeModelTest::test_method()
{
    try
    {
        int n_stations = 10;
        oskar_TelescopeModel* tel_cpu = new oskar_TelescopeModel(OSKAR_DOUBLE,
                OSKAR_LOCATION_CPU, n_stations);

        // Fill the telescope structure.

        // Copy telescope structure to GPU.
        oskar_TelescopeModel* tel_gpu = new oskar_TelescopeModel(tel_cpu,
                OSKAR_LOCATION_GPU);

        // Delete the old CPU structure.
        delete tel_cpu;

        // Copy the telescope structure back to the CPU.
        tel_cpu = new oskar_TelescopeModel(tel_gpu, OSKAR_LOCATION_CPU);

        // Delete the old GPU structure.
        delete tel_gpu;

        // Check the contents of the CPU structure.

        // Delete the CPU structure.
        delete tel_cpu;
    }
    catch (const char* msg)
    {
        CPPUNIT_FAIL(msg);
    }
}
