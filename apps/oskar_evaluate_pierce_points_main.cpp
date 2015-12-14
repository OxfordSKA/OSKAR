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

#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_version_string.h>

#include <apps/lib/oskar_evaluate_station_pierce_points.h>
#include <apps/lib/oskar_OptionParser.h>

#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv)
{
    oskar_OptionParser opt("oskar_evaulate_pierce_points",
            oskar_version_string());
    opt.addRequired("settings file");
    if (!opt.check_options(argc, argv))
        return OSKAR_FAIL;

    const char* filename = opt.getArg();

    // Create the log.
    oskar_Log* log = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_STATUS);
    oskar_log_list(log, 'M', 0, "Running binary %s", argv[0]);

    int error = OSKAR_SUCCESS;
    try
    {
        error = oskar_evaluate_station_pierce_points(filename, log);
    }
    catch (int code)
    {
        error = code;
    }

    // Check for errors.
    if (error)
        oskar_log_error(log, "Run failed: %s.", oskar_get_error_string(error));
    oskar_log_free(log);

    return error;
}

