/*
 * Copyright (c) 2012, The University of Oxford
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

#include "oskar_global.h"
#include <QtCore/QtCore>
#include <cstdio>
#include <apps/lib/oskar_OptionParser.h>

int main(int argc, char** argv)
{
    oskar_OptionParser opt("oskar_settings_get");
    opt.addRequired("settings file");
    opt.addRequired("key");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    const char* filename = opt.getArg(0);
    const char* key      = opt.getArg(1);

    if (!QFile::exists(filename))
    {
        opt.error("Specified settings file '%s' not found.", filename);
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    // Get the value.
    QSettings settings(QString(filename), QSettings::IniFormat);
    QByteArray value = settings.value(QString(key)).toByteArray();

    // Print the value.
    printf("%s\n", value.constData());

    return OSKAR_SUCCESS;
}
