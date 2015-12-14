/*
 * Copyright (c) 2013, The University of Oxford
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
void mexFunction(int num_out, mxArray** out,
        int num_in, const mxArray** in)
{
    // Parse inputs.
    if (num_in != 2 || num_out > 1) {
        oskar_matlab_usage("[value]", "settings", "get", "<file>, <key>",
                "Get the value of the specified key in the specified file");
    }

    char* filename  = mxArrayToString(in[0]);
    char* key       = mxArrayToString(in[1]);

    // Set the values.
    QSettings settings(QString(filename), QSettings::IniFormat);
    if (!settings.contains(QString(key))) {
        mexErrMsgIdAndTxt("OSKAR:ERROR",
                "ERROR: Unable to find specified key.\n");
    }
    QString value = settings.value(QString(key)).toString();

    out[0] = mxCreateString(value.toLatin1().constData());

    // Free string arrays.
    mxFree(filename);
    mxFree(key);
}
