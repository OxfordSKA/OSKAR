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

#include "apps/lib/oskar_set_up_sky.h"
#include "apps/lib/oskar_SettingsSky.h"

#include <cstdio>
#include <cstdlib>
#include <QtCore/QByteArray>

extern "C"
oskar_SkyModel* oskar_set_up_sky(const oskar_Settings& settings)
{
	int type, err;

	// Create empty sky model.
    type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_SkyModel *sky = new oskar_SkyModel(type, OSKAR_LOCATION_CPU);

    // Load sky file if it exists.
    QByteArray sky_file = settings.sky().sky_file().toAscii();
    if (!sky_file.isEmpty())
    {
    	err = sky->load(sky_file);
    	if (err)
    	{
    		delete sky;
    		return NULL;
    	}
    }


    // Compute source direction cosines relative to phase centre.
    err = sky->compute_relative_lmn(settings.obs().ra0_rad(),
            settings.obs().dec0_rad());
    if (err)
    {
    	delete sky;
    	return NULL;
    }

    // Print summary data.
    printf("\n");
    printf("= Sky model\n");
    printf("  - Sky file               = %s\n", sky_file.constData());
    printf("  - Num. sources           = %u\n", sky->num_sources);
    printf("\n");

    // Check if sky model contains no sources.
    if (sky->num_sources == 0)
    {
    	fprintf(stderr, "ERROR: Sky model contains no sources.\n");
    	delete sky;
    	return NULL;
    }

    // Return the structure.
    return sky;
}
