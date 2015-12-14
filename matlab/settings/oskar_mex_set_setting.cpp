/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <QtCore/QSettings>
#include <QtCore/QString>

#include "matlab/common/oskar_matlab_common.h"
#include <mex.h>

// oskar.settings.set(filename, key, value)
void mexFunction(int /*num_out*/, mxArray** /*out*/,
        int num_in, const mxArray** in)
{
    // Parse inputs.
    if (num_in < 3 || num_in > 4)
    {
        oskar_matlab_usage(NULL, "settings", "set", "<file>, <key>, <value>, "
                "[verbose=false]", "Set the value of the specified key "
                        "in the specified file.");
    }

    // Get inputs from MATLAB
    // Check: Key and value must be string type.
    if (!(mxIsChar(in[1]) && mxIsChar(in[2])))
    {
        oskar_matlab_error("Both the key and value must be of string type");
    }
    char* filename  = mxArrayToString(in[0]);
    char* key       = mxArrayToString(in[1]);
    char* value     = mxArrayToString(in[2]);
    bool verbose = false;
    //mexPrintf("is bool? %s\n", mxIsLogical(in[3]) ? "true" : "false");
    if (num_in == 4 && mxIsLogical(in[3])) {
        verbose = mxGetLogicals(in[3])[0];
    }
    if (verbose) {
        mexPrintf("File: %s\n", filename);
        mexPrintf("    %s=%s\n", key, value);
    }

    // Set the values.
    QSettings settings(QString(filename), QSettings::IniFormat);
    settings.setValue(QString(key), QString(value));
    settings.sync();

    // Free string arrays.
    mxFree(filename);
    mxFree(key);
    mxFree(value);
}
