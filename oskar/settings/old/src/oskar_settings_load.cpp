/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include "oskar_settings_load.h"
#include "oskar_settings_load_ionosphere.h"
#include "oskar_settings_load_observation.h"

#include "log/oskar_log.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include <QtCore/QSettings>
#include <QtCore/QVariant>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_settings_old_load(oskar_Settings_old* s, oskar_Log* log,
        const char* filename, int* status)
{
    /* Check if the settings file exists! */
    FILE* f;
    f = fopen(filename, "rb");
    if (!f)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    fclose(f);

    /* Initialise the settings arrays. */
    memset((void*) s, 0, sizeof(oskar_Settings_old));

    /* Load observation settings first as these can be the most error-prone. */
    oskar_settings_load_observation(&s->obs, log, filename, status);
    if (*status) return;
    {
        QSettings qs(QString(filename), QSettings::IniFormat);
        qs.beginGroup("simulator");
        s->sim.double_precision = qs.value("double_precision", true).toBool();
        s->sim.keep_log_file = qs.value("keep_log_file", false).toBool();
    }
    /* oskar_settings_load_ionosphere(&s->ionosphere, filename, status); */
}

#ifdef __cplusplus
}
#endif
