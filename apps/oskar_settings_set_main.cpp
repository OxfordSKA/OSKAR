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

#include <oskar_global.h>
#include <oskar_version.h>
#include <QtCore/QSettings>
#include <cstdio>
#include <apps/lib/oskar_OptionParser.h>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char** argv)
{
    oskar_OptionParser opt("oskar_settings_set");
    opt.addRequired("settings file");
    opt.addRequired("key");
    opt.addOptional("value");
    opt.addFlag("-q", "Suppress printing", false, "--quiet");
    if (!opt.check_options(argc, argv)) return OSKAR_ERR_INVALID_ARGUMENT;

    vector<string> args = opt.getArgs();
    int num_args = args.size();

    const char* filename = args[0].c_str();
    const char* key = args[1].c_str();

    const char* value    = opt.getArg(2);
    bool quiet = opt.isSet("-q") ? true : false;

    if (!quiet) {
        printf("File: %s\n", filename);
        printf("    %s=%s\n", key, value);
    }

    // Set the value.
    QSettings settings(QString(filename), QSettings::IniFormat);
    if (!settings.contains("version"))
        settings.setValue("version", OSKAR_VERSION_STR);
    if (num_args == 3) {
        settings.setValue(QString(key), QString(value));
    }
    else {
        settings.remove(QString(key));
    }

    return OSKAR_SUCCESS;
}
