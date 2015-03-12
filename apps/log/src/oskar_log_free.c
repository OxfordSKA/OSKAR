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

#include <private_log.h>
#include <oskar_version.h>
#include <oskar_log.h>
#include <oskar_log_file_exists.h>
#include <oskar_log_system_clock_string.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_log_free(oskar_Log* log)
{
    /* Print closing message. */
    oskar_log_section(log, 'M', "OSKAR-%s ending at %s.",
            OSKAR_VERSION_STR, oskar_log_system_clock_string(0));

    /* If log is NULL, there's nothing more to do. */
    if (!log) return;

    /* Close the file. */
    if (log->file) fclose(log->file);

    /* If flag is set to delete, then remove the log file. */
    if (!log->keep_file && log->name)
    {
        if (oskar_log_file_exists(log->name))
            remove(log->name);
    }

    /* Free memory for the log data. */
    free(log->name);
    free(log->code);
    free(log->offset);
    free(log->length);
    free(log->timestamp);

#ifdef _OPENMP
    omp_destroy_lock(&log->mutex);
#endif

    /* Free the structure itself. */
    free(log);
}

#ifdef __cplusplus
}
#endif
