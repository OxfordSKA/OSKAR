/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "log/private_log.h"
#include "log/oskar_log.h"
#include "oskar_version.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OSKAR_OS_WIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

oskar_Log* oskar_log_create(int file_priority, int term_priority)
{
    oskar_Log* log = 0;
    char fname1[64], fname2[64];
    char* current_dir = 0;
    size_t buf_len = 0;
    int time_data[10];
    int i = 0, n = 0;

    /* Allocate the structure. */
    log = (oskar_Log*) malloc(sizeof(oskar_Log));
    if (!log) return 0;

    /* Initialise memory for the log data. */
    log->keep_file = 1;
    log->file_priority = file_priority;
    log->term_priority = term_priority;
    log->name = 0;
    log->code = 0;
    log->offset = 0;
    log->length = 0;
    log->timestamp = 0;
    log->size = 0;
    log->capacity = 0;
    log->file = 0;
    log->value_width = OSKAR_LOG_DEFAULT_VALUE_WIDTH;

    /* Get the system time information. */
    oskar_log_system_clock_data(0, time_data);

    /* Construct log file name root. */
#if __STDC_VERSION__ >= 199901L
    n = snprintf(fname1, sizeof(fname1),
            "oskar_%.4d-%.2d-%.2d_%.2d%.2d%.2d",
            time_data[5], time_data[4], time_data[3],
            time_data[2], time_data[1], time_data[0]);
#else
    n = sprintf(fname1,
            "oskar_%.4d-%.2d-%.2d_%.2d%.2d%.2d",
            time_data[5], time_data[4], time_data[3],
            time_data[2], time_data[1], time_data[0]);
#endif
    if (n < 0 || n >= (int)sizeof(fname1))
        return 0;

    /* Construct a unique log file name. */
    do
    {
        ++i;
        if (i == 1)
        {
#if __STDC_VERSION__ >= 199901L
            n = snprintf(fname2, sizeof(fname2), "%s.log", fname1);
#else
            n = sprintf(fname2, "%s.log", fname1);
#endif
        }
        else
        {
#if __STDC_VERSION__ >= 199901L
            n = snprintf(fname2, sizeof(fname2), "%s_%d.log", fname1, i);
#else
            n = sprintf(fname2, "%s_%d.log", fname1, i);
#endif
        }
        if (n < 0 || n >= (int)sizeof(fname1))
            return 0;
    }
    while (oskar_log_file_exists(fname2));

    /* Open the log file for appending, and save the file handle. */
    log->file = fopen(fname2, "a+");

    /* Write standard header. */
    oskar_log_section(log, 'M', "OSKAR-%s starting at %s.",
            OSKAR_VERSION_STR, oskar_log_system_clock_string(0));

    /* Get the current working directory. */
#ifdef OSKAR_OS_WIN
    buf_len = GetCurrentDirectory(0, NULL);
    current_dir = (char*) calloc(buf_len, sizeof(char));
    GetCurrentDirectory((DWORD) buf_len, current_dir);
#else
    do
    {
        buf_len += 256;
        current_dir = (char*) realloc(current_dir, buf_len);
    }
    while (getcwd(current_dir, buf_len) == NULL);
#endif
    oskar_log_message(log, 'M', 0, "Current dir is %s", current_dir);
    free(current_dir);

    /* Write a message to say if the log file was opened successfully. */
    if (log->file)
    {
        /* Save the file name in the structure. */
        log->name = (char*) calloc(n + 1, sizeof(char));
        strcpy(log->name, fname2);
        oskar_log_message(log, 'M', 0, "Logging to file %s", fname2);
    }
    else
    {
        oskar_log_warning(log, "File error: log file could not be created.");
    }

    return log;
}

#ifdef __cplusplus
}
#endif
