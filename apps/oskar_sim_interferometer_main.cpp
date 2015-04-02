/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <apps/lib/oskar_sim_interferometer.h>
#include <apps/lib/oskar_OptionParser.h>

#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_version_string.h>

#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv)
{
    int error = 0;

    oskar_OptionParser opt("oskar_sim_interferometer", oskar_version_string());
    opt.addRequired("settings file");
    opt.addFlag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return OSKAR_ERR_INVALID_ARGUMENT;

    // Create the log.
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = opt.isSet("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
    oskar_Log* log = oskar_log_create(file_priority, term_priority);

    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);

    // Run simulation.
    oskar_sim_interferometer(opt.getArg(0), log, &error);

    // Check for errors.
    if (error)
    {
        oskar_log_error(log, "Run failed with code %i: %s.", error,
                oskar_get_error_string(error));
    }
    oskar_log_free(log);

    return error;
}
